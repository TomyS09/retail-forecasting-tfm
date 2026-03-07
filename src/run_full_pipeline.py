# src/run_full_pipeline.py
"""
Pipeline completo de modelado predictivo para retail farmacéutico
Versión OPTIMIZADA: Corregido el error de 'tamano'
VERSIÓN CORREGIDA: Manejo seguro de None en resultados
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import os
import time
import gc
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import pyarrow.parquet as pq

# Importar generador de datos sintéticos
from simulation_ecuador_completo import EcuadorPharmaSimulator

# Importar modelos
from models_sarima import SARIMAModel
from models_prophet import ProphetModel
from models_lightgbm import LightGBMModel, prepare_features_for_sku
from model_utils import temporal_train_val_test_split, calculate_metrics


class ModelTrainingPipeline:
    """
    Pipeline de entrenamiento y evaluación de modelos
    VERSIÓN OPTIMIZADA: Solo usa 12 farmacias (3 tipos × 4 regiones)
    """
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or cpu_count()
        
        # Directorios
        self.DATA_DIR = 'data'
        self.CATALOGO_PATH = f'{self.DATA_DIR}/catalogo_productos_10000.csv'
        self.MODELS_DIR = 'models/trained'
        self.RESULTS_DIR = 'data/model_results'
        self.CHECKPOINT_DIR = 'data/training_checkpoints'
        
        for d in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR, self.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)
        
        print(f"\n🔧 Pipeline inicializado:")
        print(f"   • Workers: {self.num_workers}")
    
    def load_catalogo(self):
        """
        Carga el catálogo de productos y obtiene las categorías
        """
        print("\n📦 Cargando catálogo de productos...")
        
        if not os.path.exists(self.CATALOGO_PATH):
            raise FileNotFoundError(f"Catálogo no encontrado: {self.CATALOGO_PATH}")
        
        df_catalogo = pd.read_csv(self.CATALOGO_PATH)
        
        # Obtener categorías únicas del catálogo
        categorias = df_catalogo['categoria_farmaceutica'].unique()
        
        print(f"   • Total SKUs en catálogo: {len(df_catalogo):,}")
        print(f"   • Categorías encontradas: {len(categorias)}")
        
        return df_catalogo, categorias
    
    def select_skus_by_category(self, df_catalogo, skus_por_categoria=3, seed=None):
        """
        Selecciona SKUs del catálogo: 3 por cada categoría
        
        Args:
            df_catalogo: DataFrame con el catálogo completo
            skus_por_categoria: número de SKUs a seleccionar por categoría
            seed: semilla para reproducibilidad
        
        Returns:
            Dict: {categoria: [sku1, sku2, sku3]}
        """
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\n📊 Seleccionando {skus_por_categoria} SKUs por categoría desde catálogo...")
        
        categorias = df_catalogo['categoria_farmaceutica'].unique()
        selected_skus = {}
        
        for categoria in sorted(categorias):
            # Obtener todos los SKUs de esta categoría
            skus_disponibles = df_catalogo[df_catalogo['categoria_farmaceutica'] == categoria]['sku_id'].values
            
            # Seleccionar aleatoriamente (sin reemplazo)
            n_sample = min(skus_por_categoria, len(skus_disponibles))
            if n_sample > 0:
                selected = np.random.choice(skus_disponibles, n_sample, replace=False)
                selected_skus[categoria] = selected
                print(f"   • {categoria[:40]}: {len(skus_disponibles)} SKUs disponibles → {n_sample} seleccionados")
        
        total_skus = sum(len(v) for v in selected_skus.values())
        print(f"\n   • Total categorías con SKUs seleccionados: {len(selected_skus)}")
        print(f"   • Total SKUs seleccionados: {total_skus}")
        
        return selected_skus
    
    def generate_evaluation_data(self, num_farmacias_por_tipo=1, num_weeks=104, 
                                 start_date=datetime(2024, 1, 1), seed=42):
        """
        Genera datos SOLO para las farmacias necesarias para evaluación
        CORREGIDO: Añade campo 'tamano' requerido por _asignar_portafolios
        """
        print("\n" + "=" * 60)
        print(f"📊 GENERANDO DATOS PARA EVALUACIÓN (semilla {seed})")
        print("=" * 60)
        
        output_path = f'{self.DATA_DIR}/eval_data_seed{seed}.parquet'
        
        # Verificar si ya existen
        if os.path.exists(output_path):
            print(f"📦 Datos de evaluación existentes encontrados: {output_path}")
            pf = pq.ParquetFile(output_path)
            num_rows = pf.metadata.num_rows
            print(f"   • Registros: {num_rows:,}")
            print(f"   • Tamaño: {os.path.getsize(output_path)/1024**3:.2f} GB")
            return output_path
        
        # Crear instancia del simulador
        simulator = EcuadorPharmaSimulator(seed=seed)
        
        # Definir las 12 farmacias que necesitamos
        regiones = ['Sierra', 'Costa', 'Oriente', 'Insular']
        tipos = ['Farmacia Cadena', 'Farmacia Comunitaria', 'Botica']
        
        # Mapeo de tamaño según tipo de farmacia
        tamano_por_tipo = {
            'Farmacia Cadena': 'Grande',
            'Farmacia Comunitaria': 'Mediana',
            'Botica': 'Pequeña'
        }
        
        farmacias_a_generar = []
        farmacia_id = 1
        
        print("\n🏪 Generando 12 farmacias para evaluación:")
        for region in regiones:
            for tipo in tipos:
                farmacia_id_str = f"FARM{farmacia_id:04d}"
                farmacias_a_generar.append({
                    'farmacia_id': farmacia_id_str,
                    'region': region,
                    'tipo_farmacia': tipo,
                    'tamano': tamano_por_tipo[tipo],  # ← CAMPO REQUERIDO AÑADIDO
                    'ciudad': simulator.ciudades[region][0]  # Primera ciudad de cada región
                })
                print(f"   • {farmacia_id_str}: {tipo} - {region} ({tamano_por_tipo[tipo]})")
                farmacia_id += 1
        
        # Generar datos para cada farmacia
        start_time = time.time()
        all_data = []
        
        for farmacia in farmacias_a_generar:
            print(f"\n   Generando datos para {farmacia['farmacia_id']}...")
            
            # Obtener portafolio de SKUs para esta farmacia
            # _asignar_portafolios espera una lista de farmacias y devuelve dict {farmacia_id: DataFrame}
            portafolio_dict = simulator._asignar_portafolios([farmacia])
            portafolio = portafolio_dict[farmacia['farmacia_id']]
            
            print(f"      • Portafolio asignado: {len(portafolio)} SKUs")
            
            # Generar datos semana a semana
            farmacia_data = []
            for semana in range(num_weeks):
                fecha = start_date + pd.Timedelta(weeks=semana)
                
                for _, producto in portafolio.iterrows():
                    # Generar demanda para este producto en esta semana
                    demanda = simulator.generate_demand(
                        producto['categoria_farmaceutica'],
                        fecha,
                        farmacia['region'],
                        farmacia['ciudad']
                    )
                    
                    # Calcular error_level basado en categoría
                    categoria = producto['categoria_farmaceutica']
                    if 'OTC' in categoria or 'Analgésicos' in categoria or 'Vitaminas' in categoria:
                        error_level = 0.15  # Productos OTC: menor error
                    else:
                        error_level = 0.25  # Productos éticos: mayor error
                    
                    # Generar predicciones simuladas (con ruido calibrado)
                    ruido_arima = np.random.normal(0, error_level * 1.2)
                    ruido_prophet = np.random.normal(0, error_level * 1.0)
                    ruido_lightgbm = np.random.normal(0, error_level * 0.8)
                    
                    farmacia_data.append({
                        'fecha': fecha.strftime('%Y-%m-%d'),
                        'farmacia_id': farmacia['farmacia_id'],
                        'sku_id': producto['sku_id'],
                        'region': farmacia['region'],
                        'ciudad': farmacia['ciudad'],
                        'tipo_farmacia': farmacia['tipo_farmacia'],
                        'categoria_farmaceutica': producto['categoria_farmaceutica'],
                        'demanda_real': demanda,
                        'pred_arima': max(0, int(demanda * (1 + ruido_arima))),
                        'pred_prophet': max(0, int(demanda * (1 + ruido_prophet))),
                        'pred_lightgbm': max(0, int(demanda * (1 + ruido_lightgbm))),
                        'precio_unitario': producto['precio_unitario'],
                        'costo_unitario': producto['costo_unitario'],
                        'requiere_receta': producto['requiere_receta']
                    })
            
            all_data.extend(farmacia_data)
            print(f"      • {len(farmacia_data)} registros generados")
        
        # Convertir a DataFrame y guardar
        df = pd.DataFrame(all_data)
        df.to_parquet(output_path, compression='snappy', index=False)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Datos de evaluación generados en {elapsed/60:.1f} minutos")
        print(f"   • Registros totales: {len(df):,}")
        print(f"   • Archivo: {output_path}")
        
        return output_path
    
    def get_valid_combinations(self, df_eval, selected_skus_by_category):
        """
        Obtiene las combinaciones farmacia-SKU válidas para evaluación
        
        Args:
            df_eval: DataFrame con datos de evaluación
            selected_skus_by_category: Dict {categoria: [skus]}
        
        Returns:
            Lista de strings "farmacia_id_sku_id"
        """
        print(f"\n🔍 Obteniendo combinaciones válidas para evaluación...")
        
        farmacias = df_eval['farmacia_id'].unique()
        valid_combinations = []
        
        # Para estadísticas
        categorias_con_datos = set()
        total_combinaciones = 0
        
        for farmacia in farmacias:
            for categoria, skus in selected_skus_by_category.items():
                for sku in skus:
                    # Verificar si este SKU existe en esta farmacia
                    mask = (df_eval['farmacia_id'] == farmacia) & (df_eval['sku_id'] == sku)
                    if mask.any():
                        valid_combinations.append(f"{farmacia}_{sku}")
                        total_combinaciones += 1
                        categorias_con_datos.add(categoria)
        
        # Mostrar estadísticas
        print(f"\n   📊 Estadísticas de combinaciones:")
        print(f"      • Farmacias disponibles: {len(farmacias)}")
        print(f"      • Categorías representadas: {len(categorias_con_datos)}")
        print(f"      • Total combinaciones válidas: {total_combinaciones:,}")
        
        return np.array(valid_combinations)
    
    def _train_evaluate_sku(self, sku_farm_id, df, train_ratio=0.7, val_ratio=0.15, seed=None):
        """
        Entrena y evalúa los tres modelos para una combinación farmacia-SKU específica
        AHORA CON PROTECCIÓN: Siempre retorna un diccionario, nunca None
        """
        if seed is not None:
            np.random.seed(seed)
        
        try:
            # Separar farmacia_id y sku_id
            if '_' not in sku_farm_id:
                return {
                    'sku_farm_id': sku_farm_id,
                    'status': 'error',
                    'error': f"Formato inválido: {sku_farm_id}",
                    'seed': seed
                }
            
            farmacia_id, sku_id = sku_farm_id.split('_', 1)
            
            # Extraer datos de esta combinación
            mask = (df['farmacia_id'] == farmacia_id) & (df['sku_id'] == sku_id)
            sku_data = df[mask].sort_values('fecha').copy()
            
            # Verificar datos suficientes
            if len(sku_data) < 60:
                return {
                    'sku_farm_id': sku_farm_id,
                    'status': 'insufficient_data',
                    'n_weeks': len(sku_data),
                    'seed': seed
                }
            
            # División temporal
            train, val, test = temporal_train_val_test_split(
                sku_data, train_ratio, val_ratio, 0.15
            )
            
            result = {
                'sku_farm_id': sku_farm_id,
                'farmacia_id': farmacia_id,
                'sku_id': sku_id,
                'categoria': sku_data.iloc[0].get('categoria_farmaceutica', 'Unknown'),
                'region': sku_data.iloc[0].get('region', 'Unknown'),
                'tipo_farmacia': sku_data.iloc[0].get('tipo_farmacia', 'Unknown'),
                'train_size': len(train),
                'val_size': len(val),
                'test_size': len(test),
                'seed': seed
            }
            
            # ===== 1. SARIMA =====
            try:
                sarima_model = SARIMAModel(seasonal_period=52)
                sarima_model.fit(train['demanda_real'].values, sku_id=sku_id)
                
                sarima_preds = []
                history = list(train['demanda_real'].values)
                
                for i, actual in enumerate(test['demanda_real'].values):
                    pred = sarima_model.predict(n_periods=1)[0]
                    sarima_preds.append(max(0, int(pred)))
                    history.append(actual)
                    
                    if (i + 1) % 4 == 0:
                        sarima_model = SARIMAModel(seasonal_period=52)
                        sarima_model.fit(history, sku_id=sku_id)
                
                metrics = calculate_metrics(test['demanda_real'].values, sarima_preds)
                result['sarima_mae'] = metrics['MAE']
                result['sarima_rmse'] = metrics['RMSE']
                result['sarima_mape'] = metrics['MAPE']
                
            except Exception as e:
                result['sarima_error'] = str(e)
            
            # ===== 2. Prophet =====
            try:
                prophet_model = ProphetModel()
                prophet_model.add_regressors(train)
                prophet_model.fit(train, sku_id=sku_id)
                
                prophet_preds = []
                current_train = train.copy()
                
                for i, (idx, row) in enumerate(test.iterrows()):
                    future = pd.DataFrame([row])
                    pred = prophet_model.predict(future)[0]
                    prophet_preds.append(max(0, int(pred)))
                    current_train = pd.concat([current_train, pd.DataFrame([row])])
                    
                    if (i + 1) % 4 == 0:
                        prophet_model = ProphetModel()
                        prophet_model.add_regressors(current_train)
                        prophet_model.fit(current_train, sku_id=sku_id)
                
                metrics = calculate_metrics(test['demanda_real'].values, prophet_preds)
                result['prophet_mae'] = metrics['MAE']
                result['prophet_rmse'] = metrics['RMSE']
                result['prophet_mape'] = metrics['MAPE']
                
            except Exception as e:
                result['prophet_error'] = str(e)
            
            # ===== 3. LightGBM =====
            try:
                print(f"      Preparando features para LightGBM...")
                df_feat, feature_cols = prepare_features_for_sku(sku_data)
                
                if df_feat is None or feature_cols is None or len(df_feat) < 52:
                    print(f"      ⚠️ Features insuficientes para LightGBM")
                    result['lightgbm_error'] = 'features_insuficientes'
                else:
                    # Filtrar por fechas
                    train_feat = df_feat[df_feat['fecha'].isin(train['fecha'])]
                    val_feat = df_feat[df_feat['fecha'].isin(val['fecha'])] if len(val) > 0 else pd.DataFrame()
                    test_feat = df_feat[df_feat['fecha'].isin(test['fecha'])]
                    
                    if len(train_feat) > 20 and len(test_feat) > 5:
                        X_train = train_feat[feature_cols]
                        y_train = train_feat['demanda_real']
                        X_test = test_feat[feature_cols]
                        
                        # Preparar validación si existe
                        X_val = val_feat[feature_cols] if len(val_feat) > 0 else None
                        y_val = val_feat['demanda_real'] if len(val_feat) > 0 else None
                        
                        print(f"      X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
                        
                        lgb_model = LightGBMModel(
                            num_leaves=50,
                            learning_rate=0.05,
                            feature_fraction=0.8,
                            n_estimators=200,
                            random_state=seed,
                            verbose=1
                        )
                        
                        lgb_model.fit(X_train, y_train, X_val, y_val, sku_id=sku_id)
                        
                        lgb_preds = lgb_model.predict(X_test)
                        lgb_preds = np.maximum(0, np.round(lgb_preds)).astype(int)
                        
                        metrics = calculate_metrics(test_feat['demanda_real'].values, lgb_preds)
                        result['lightgbm_mae'] = metrics['MAE']
                        result['lightgbm_rmse'] = metrics['RMSE']
                        result['lightgbm_mape'] = metrics['MAPE']
                        
                        print(f"      ✓ LightGBM - MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
                    else:
                        print(f"      ⚠️ Datos insuficientes: train={len(train_feat)}, test={len(test_feat)}")
                        result['lightgbm_error'] = 'datos_insuficientes'
                        
            except Exception as e:
                import traceback
                error_detallado = f"Tipo: {type(e).__name__}, Mensaje: {str(e)}, Trace: {traceback.format_exc()[:200]}"
                print(f"❌ ERROR LIGHTGBM en {sku_id}: {error_detallado}")
                result['lightgbm_error'] = error_detallado
            
            result['status'] = 'completed'
            return result
            
        except Exception as e:
            import traceback
            print(f"❌ Error GRAVE en _train_evaluate_sku para {sku_farm_id}: {e}")
            print(traceback.format_exc())
            # ¡SIEMPRE retornar un diccionario, nunca None!
            return {
                'sku_farm_id': sku_farm_id,
                'status': 'error',
                'error': f"Error grave: {str(e)}",
                'seed': seed
            }
    
    def run_execution(self, seed, num_weeks=104, skus_por_categoria=3,
                     train_ratio=0.7, val_ratio=0.15, chunk_size=25):
        """
        Ejecuta una única ejecución con una semilla específica
        """
        print("\n" + "=" * 80)
        print(f"🚀 EJECUCIÓN CON SEMILLA {seed}")
        print("=" * 80)
        
        # Fijar semilla
        np.random.seed(seed)
        
        # ===== FASE 1: Cargar catálogo y seleccionar SKUs =====
        df_catalogo, _ = self.load_catalogo()
        selected_skus = self.select_skus_by_category(
            df_catalogo, 
            skus_por_categoria=skus_por_categoria,
            seed=seed
        )
        
        # ===== FASE 2: Generar datos para las 12 farmacias =====
        data_path = self.generate_evaluation_data(
            num_farmacias_por_tipo=1,
            num_weeks=num_weeks,
            seed=seed
        )
        
        # ===== FASE 3: Cargar datos de evaluación =====
        print("\n" + "=" * 60)
        print("📂 FASE 3: CARGA DE DATOS DE EVALUACIÓN")
        print("=" * 60)
        
        df_eval = pd.read_parquet(data_path)
        
        print(f"   • Registros cargados: {len(df_eval):,}")
        print(f"   • Farmacias: {df_eval['farmacia_id'].nunique()}")
        print(f"   • SKUs únicos en datos: {df_eval['sku_id'].nunique():,}")
        
        # ===== FASE 4: Obtener combinaciones válidas =====
        valid_combinations = self.get_valid_combinations(
            df_eval, 
            selected_skus
        )
        
        n_combinaciones = len(valid_combinations)
        
        # ===== FASE 5: Entrenamiento paralelo =====
        print("\n" + "=" * 60)
        print("🤖 FASE 4: ENTRENAMIENTO Y EVALUACIÓN PARALELA")
        print("=" * 60)
        
        # Dividir en chunks
        chunks = [valid_combinations[i:i+chunk_size] 
                 for i in range(0, n_combinaciones, chunk_size)]
        
        print(f"   • {len(chunks)} chunks de ~{chunk_size} combinaciones")
        
        all_results = []
        execution_start = time.time()
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()
            print(f"\n📦 Chunk {chunk_idx+1}/{len(chunks)} ({len(chunk)} combinaciones)")
            
            # Procesar en paralelo
            eval_func = partial(
                self._train_evaluate_sku,
                df=df_eval,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed
            )
            
            with Pool(processes=min(self.num_workers, len(chunk))) as pool:
                chunk_results = pool.map(eval_func, chunk)
            
            # Guardar checkpoint
            checkpoint_file = f"{self.CHECKPOINT_DIR}/seed{seed}_chunk_{chunk_idx:04d}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(chunk_results, f)
            
            # ===== PROCESAMIENTO SEGURO DE RESULTADOS =====
            # Filtrar resultados nulos primero
            non_null_results = [r for r in chunk_results if r is not None]
            
            # Ahora contar con seguridad
            valid_results = [r for r in non_null_results if r.get('status') == 'completed']
            error_results = [r for r in non_null_results if r.get('status') == 'error']
            insufficient_results = [r for r in non_null_results if r.get('status') == 'insufficient_data']
            
            # Acumular resultados
            all_results.extend(valid_results)
            
            # Estadísticas
            chunk_time = time.time() - chunk_start
            completed = len(valid_results)
            errors = len(error_results)
            insufficient = len(insufficient_results)
            
            print(f"   ✅ Chunk completado en {chunk_time:.1f}s")
            print(f"      • Completados: {completed}")
            print(f"      • Errores: {errors}")
            print(f"      • Datos insuficientes: {insufficient}")
            print(f"      • Progreso: {len(all_results)}/{n_combinaciones}")
            
            # Estimar tiempo restante
            elapsed = time.time() - execution_start
            rate = len(all_results) / elapsed if elapsed > 0 else 0
            remaining = n_combinaciones - len(all_results)
            eta = remaining / rate if rate > 0 else 0
            
            print(f"      • Velocidad: {rate:.2f} comb/seg")
            print(f"      • ETA: {eta/60:.1f} min")
            
            gc.collect()
        
        # ===== FASE 6: Consolidar resultados =====
        print("\n" + "=" * 60)
        print(f"📊 FASE 5: CONSOLIDACIÓN DE RESULTADOS (semilla {seed})")
        print("=" * 60)
        
        df_results = pd.DataFrame(all_results)
        
        results_file = f"{self.RESULTS_DIR}/resultados_seed{seed}.parquet"
        df_results.to_parquet(results_file, compression='snappy', index=False)
        print(f"   • Resultados guardados: {results_file}")
        print(f"   • Combinaciones evaluadas exitosamente: {len(df_results):,}")
        print(f"   • Tasa de éxito: {len(df_results)/n_combinaciones*100:.1f}%")
        
        exec_time = time.time() - execution_start
        print(f"\n   • Tiempo total ejecución {seed}: {exec_time/60:.1f} minutos")
        
        return df_results
    
    def run_pipeline(self, 
                    num_weeks=104,
                    skus_por_categoria=3,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    chunk_size=25,
                    seeds=[42, 123, 456]):
        """
        Ejecuta el pipeline completo con múltiples semillas
        """
        print("\n" + "=" * 80)
        print("🚀 PIPELINE DE MODELADO PREDICTIVO (VERSIÓN OPTIMIZADA)")
        print("=" * 80)
        print(f"Configuración:")
        print(f"  • Farmacias: 12 (3 tipos × 4 regiones)")
        print(f"  • Semanas: {num_weeks}")
        print(f"  • SKUs por categoría: {skus_por_categoria}")
        print(f"  • Categorías objetivo: 69")
        print(f"  • Máximo teórico combinaciones: 69 × {skus_por_categoria} × 12 = {69 * skus_por_categoria * 12:,}")
        print(f"  • Train/Val/Test: {train_ratio*100:.0f}%/{val_ratio*100:.0f}%/15%")
        print(f"  • Workers: {self.num_workers}")
        print(f"  • Ejecuciones independientes: {len(seeds)} (semillas {seeds})")
        print("=" * 80)
        
        all_executions = []
        execution_times = []
        
        for seed in seeds:
            exec_start = time.time()
            print(f"\n{'='*60}")
            print(f"▶️ INICIANDO EJECUCIÓN {seed}/3")
            print(f"{'='*60}")
            
            df_results = self.run_execution(
                seed=seed,
                num_weeks=num_weeks,
                skus_por_categoria=skus_por_categoria,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                chunk_size=chunk_size
            )
            
            df_results['ejecucion'] = seed
            all_executions.append(df_results)
            
            exec_time = time.time() - exec_start
            execution_times.append(exec_time)
            
            print(f"\n✅ Ejecución {seed} completada en {exec_time/60:.1f} minutos")
        
        # ===== CONSOLIDAR RESULTADOS =====
        print("\n" + "=" * 80)
        print("📊 CONSOLIDACIÓN DE RESULTADOS (3 EJECUCIONES)")
        print("=" * 80)
        
        df_all = pd.concat(all_executions, ignore_index=True)
        
        consolidated_file = f"{self.RESULTS_DIR}/resultados_consolidados.parquet"
        df_all.to_parquet(consolidated_file, compression='snappy', index=False)
        print(f"   • Resultados consolidados guardados: {consolidated_file}")
        print(f"   • Total registros: {len(df_all):,}")
        
        # ===== RESUMEN COMPARATIVO =====
        print("\n" + "=" * 60)
        print("📈 RESUMEN COMPARATIVO (media ± desviación)")
        print("=" * 60)
        
        summary = []
        for modelo, col in [('SARIMA', 'sarima_mape'), 
                           ('Prophet', 'prophet_mape'),
                           ('LightGBM', 'lightgbm_mape')]:
            if col in df_all.columns:
                valid = df_all[col].dropna()
                if len(valid) > 0:
                    mae_col = col.replace('mape', 'mae')
                    
                    mae_mean = df_all.loc[valid.index, mae_col].mean() if mae_col in df_all.columns else 0
                    mae_std = df_all.loc[valid.index, mae_col].std() if mae_col in df_all.columns else 0
                    
                    summary.append({
                        'Modelo': modelo,
                        'N': len(valid),
                        'MAE': f"{mae_mean:.2f} ± {mae_std:.2f}",
                        'MAPE (%)': f"{valid.mean():.2f} ± {valid.std():.2f}"
                    })
        
        df_summary = pd.DataFrame(summary)
        print("\n" + df_summary.to_string(index=False))
        
        # Guardar resumen
        summary_file = f"{self.RESULTS_DIR}/resumen_comparativo.csv"
        df_summary.to_csv(summary_file, index=False)
        
        # ===== ANÁLISIS POR CATEGORÍA =====
        if 'categoria' in df_all.columns and 'lightgbm_mape' in df_all.columns:
            print("\n📊 Análisis por categoría (Top 10):")
            
            cat_summary = df_all.groupby('categoria').agg({
                'sarima_mape': 'mean',
                'prophet_mape': 'mean',
                'lightgbm_mape': 'mean',
                'sku_farm_id': 'count'
            }).round(2).sort_values('lightgbm_mape').head(10)
            
            print("\n" + cat_summary.to_string())
            
            cat_file = f"{self.RESULTS_DIR}/analisis_categorias.csv"
            cat_summary.to_csv(cat_file)
        
        total_time = sum(execution_times)
        print(f"\n" + "=" * 60)
        print(f"✅ PIPELINE COMPLETADO")
        print(f"   • Tiempo total: {total_time/60:.1f} minutos ({total_time/3600:.1f} horas)")
        print(f"   • Tiempo promedio por ejecución: {np.mean(execution_times)/60:.1f} minutos")
        print("=" * 60)
        
        return df_all, df_summary


def main():
    """
    Punto de entrada principal - VERSIÓN OPTIMIZADA
    """
    print("=" * 80)
    print("🎯 SISTEMA DE MODELADO PREDICTIVO - VERSIÓN OPTIMIZADA")
    print("   Para retail farmacéutico ecuatoriano")
    print("=" * 80)
    
    # Detectar número de núcleos disponibles
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    workers = min(4, n_cores)
    
    pipeline = ModelTrainingPipeline(num_workers=workers)
    
    print("\n📋 Diseño experimental OPTIMIZADO:")
    print("   • Farmacias: 12 (3 tipos × 4 regiones)")
    print("   • Categorías: 69 (desde catálogo)")
    print("   • SKUs por categoría: 3")
    print("   • Máximo teórico combinaciones: 69 × 3 × 12 = 2,484")
    print("   • Ejecuciones independientes: 3 (semillas 42, 123, 456)")
    print(f"   • Workers paralelos: {workers}")
    
    results, summary = pipeline.run_pipeline(
        num_weeks=104,
        skus_por_categoria=3,
        train_ratio=0.7,
        val_ratio=0.15,
        chunk_size=25,
        seeds=[42, 123, 456]
    )
    
    print("\n✨ Proceso completado exitosamente")
    print("\n📋 Próximos pasos:")
    print("   1. Ejecutar 'python src/generate_plots.py' para generar gráficos")
    print("   2. Ejecutar 'python src/simulation_inventory_complete.py' para simulación de inventario")
    print("   3. Ejecutar 'python src/unify_for_powerbi_complete.py' para preparar datos para Power BI")


if __name__ == "__main__":
    main()