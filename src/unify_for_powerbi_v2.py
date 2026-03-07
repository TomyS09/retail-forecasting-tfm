# src/unify_for_powerbi_v2.py
"""
Unificación ULTRA RÁPIDA para Power BI - VERSIÓN DEFINITIVA
Integra:
- Resultados de las 3 ejecuciones de modelos (semillas 42,123,456)
- Demanda real desde eval_data_seed*.parquet (con fechas)
- Simulación de inventario para la muestra
- Genera 3 paneles específicos con media ± desviación
- Incluye relaciones entidad-relación en SQLite
- CORREGIDO: datos_detalle con fecha, demanda_real y semilla
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
import time
import gc
import glob
import shutil
import sqlite3
import multiprocessing as mp


class PowerBIFinalUnifier:
    """
    Unificador final para Power BI - Versión definitiva
    """
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        
        # Directorios
        self.RESULTS_DIR = 'data/model_results'
        self.INVENTORY_DIR = 'data/inventory_sample'
        self.EVAL_DIR = 'data'  # eval_data_seed*.parquet
        self.OUTPUT_DIR = 'dashboard/data_farmacias'
        self.TEMP_DIR = 'dashboard/temp'
        
        for d in [self.OUTPUT_DIR, self.TEMP_DIR]:
            os.makedirs(d, exist_ok=True)
        
        print(f"\n🔧 Unificador Power BI inicializado:")
        print(f"   • Workers: {self.num_workers}")
    
    def load_all_executions(self):
        """
        Carga y consolida las 3 ejecuciones de modelos
        """
        print("\n📊 Cargando resultados de las 3 ejecuciones...")
        
        all_dfs = []
        seeds = [42, 123, 456]
        
        for seed in seeds:
            file_path = f'{self.RESULTS_DIR}/resultados_seed{seed}.parquet'
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                df['semilla'] = seed
                all_dfs.append(df)
                print(f"   • Seed {seed}: {len(df)} registros")
            else:
                print(f"   ⚠️ No encontrado: {file_path}")
        
        if not all_dfs:
            raise FileNotFoundError("No se encontraron resultados de modelos")
        
        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"\n   ✓ Consolidado: {len(df_all):,} registros totales")
        
        return df_all
    
    def load_demand_data(self):
        """
        Carga datos de demanda real desde eval_data_seed*.parquet
        ¡ESTOS DATOS TIENEN FECHA!
        """
        print("\n📈 Cargando datos de demanda real (con fechas)...")
        
        all_demand = []
        seeds = [42, 123, 456]
        
        for seed in seeds:
            file_path = f'{self.EVAL_DIR}/eval_data_seed{seed}.parquet'
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                df['semilla'] = seed
                # Conservar TODAS las columnas necesarias
                df_demand = df[['fecha', 'farmacia_id', 'sku_id', 
                                'demanda_real', 'region', 'tipo_farmacia', 
                                'categoria_farmaceutica', 'semilla']].copy()
                all_demand.append(df_demand)
                print(f"   • Seed {seed}: {len(df_demand):,} registros de demanda")
            else:
                print(f"   ⚠️ No encontrado: {file_path}")
        
        if not all_demand:
            print("   ⚠️ No se encontraron datos de demanda real")
            return pd.DataFrame()
        
        df_all = pd.concat(all_demand, ignore_index=True)
        # Asegurar tipo fecha
        df_all['fecha'] = pd.to_datetime(df_all['fecha'])
        print(f"\n   ✓ Demanda consolidada: {len(df_all):,} registros")
        print(f"   • Rango fechas: {df_all['fecha'].min()} a {df_all['fecha'].max()}")
        
        return df_all
    
    def load_inventory_data(self):
        """
        Carga datos de inventario para las 3 semillas
        """
        print("\n📦 Cargando datos de inventario...")
        
        inventory_files = glob.glob(f'{self.INVENTORY_DIR}/inventario_seed*.parquet')
        order_files = glob.glob(f'{self.INVENTORY_DIR}/ordenes_seed*.parquet')
        
        all_inventory = []
        all_orders = []
        
        for inv_file in inventory_files:
            seed = int(inv_file.split('_seed')[-1].split('.')[0])
            df_inv = pd.read_parquet(inv_file)
            df_inv['semilla'] = seed
            all_inventory.append(df_inv)
            print(f"   • Inventario seed {seed}: {len(df_inv):,} registros")
        
        for ord_file in order_files:
            seed = int(ord_file.split('_seed')[-1].split('.')[0])
            df_ord = pd.read_parquet(ord_file)
            df_ord['semilla'] = seed
            all_orders.append(df_ord)
            print(f"   • Órdenes seed {seed}: {len(df_ord):,} registros")
        
        df_inv_all = pd.concat(all_inventory, ignore_index=True) if all_inventory else pd.DataFrame()
        df_ord_all = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
        
        return df_inv_all, df_ord_all
    
    def create_dimensions(self, df_demand):
        """
        Crea tablas de dimensiones a partir de los datos de demanda
        AHORA USA df_demand como fuente principal
        """
        print("\n📋 Creando tablas de dimensiones...")
        
        # ===== dim_productos =====
        productos = df_demand[['sku_id', 'categoria_farmaceutica']].drop_duplicates().copy()
        productos = productos.rename(columns={'categoria_farmaceutica': 'categoria'})
        
        # Extraer más detalles del catálogo si existe
        catalogo_path = 'data/catalogo_productos_10000.csv'
        if os.path.exists(catalogo_path):
            df_catalogo = pd.read_csv(catalogo_path)
            dim_productos = productos.merge(
                df_catalogo[['sku_id', 'nombre_producto', 'laboratorio', 
                            'precio_unitario', 'costo_unitario', 'requiere_receta']],
                on='sku_id',
                how='left'
            )
        else:
            dim_productos = productos.copy()
            dim_productos['nombre_producto'] = dim_productos['sku_id']
            dim_productos['laboratorio'] = 'Desconocido'
            dim_productos['precio_unitario'] = 0
            dim_productos['costo_unitario'] = 0
            dim_productos['requiere_receta'] = 0
        
        dim_productos.to_parquet(f'{self.OUTPUT_DIR}/dim_productos.parquet', index=False)
        print(f"   • dim_productos: {len(dim_productos)} SKUs")
        
        # ===== dim_tiendas =====
        dim_tiendas = df_demand[['farmacia_id', 'region', 'tipo_farmacia']].drop_duplicates().copy()
        dim_tiendas = dim_tiendas.rename(columns={'farmacia_id': 'tienda_id'})
        dim_tiendas.to_parquet(f'{self.OUTPUT_DIR}/dim_tiendas.parquet', index=False)
        print(f"   • dim_tiendas: {len(dim_tiendas)} tiendas")
        
        # ===== dim_fechas =====
        fechas = pd.to_datetime(df_demand['fecha'].unique())
        fechas = pd.Series(fechas).sort_values().reset_index(drop=True)
        
        dim_fechas = pd.DataFrame({
            'fecha': fechas,
            'anio': fechas.dt.year,
            'mes': fechas.dt.month,
            'semana': fechas.dt.isocalendar().week,
            'dia_semana': fechas.dt.dayofweek,
            'nombre_mes': fechas.dt.strftime('%B'),
            'nombre_dia': fechas.dt.strftime('%A')
        })
        dim_fechas.to_parquet(f'{self.OUTPUT_DIR}/dim_fechas.parquet', index=False)
        print(f"   • dim_fechas: {len(dim_fechas)} fechas")
        
        # ===== dim_modelos =====
        dim_modelos = pd.DataFrame({
            'modelo_id': [1, 2, 3, 4],
            'modelo': ['SARIMA', 'Prophet', 'LightGBM', 'REAL'],
            'color': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6']
        })
        dim_modelos.to_parquet(f'{self.OUTPUT_DIR}/dim_modelos.parquet', index=False)
        print(f"   • dim_modelos: {len(dim_modelos)} modelos")
    
    def create_panel1_comparativa(self, df_results, df_demand):
        """
        PANEL 1: Comparativa de modelos predictivos
        CORREGIDO: Crea datos_detalle con fecha, demanda_real y semilla
        """
        print("\n📊 PANEL 1: Creando comparativa de modelos...")
        
        panel1_dir = f'{self.OUTPUT_DIR}/panel1'
        os.makedirs(panel1_dir, exist_ok=True)
        
        # ===== 1. CREAR DATOS_DETALLE COMPLETOS =====
        print(f"   • Creando datos_detalle con {len(df_demand):,} registros de demanda...")
        
        # Tomar una muestra representativa de la demanda (para no saturar)
        # 12 farmacias × 104 semanas × ~150 SKUs = ~187,200 registros
        # Es manejable para Power BI
        df_detalle = df_demand.copy()
        
        # Añadir predicciones desde df_results (merge por SKU y farmacia)
        # Como las predicciones son agregadas por SKU, las repetiremos para cada fecha
        df_pred_agg = df_results[['farmacia_id', 'sku_id', 'semilla',
                                   'sarima_mae', 'sarima_rmse', 'sarima_mape',
                                   'prophet_mae', 'prophet_rmse', 'prophet_mape',
                                   'lightgbm_mae', 'lightgbm_rmse', 'lightgbm_mape']].copy()
        
        # Hacer merge con df_detalle
        df_detalle = df_detalle.merge(
            df_pred_agg,
            on=['farmacia_id', 'sku_id', 'semilla'],
            how='left'
        )
        
        print(f"   • datos_detalle final: {len(df_detalle):,} registros")
        print(f"   • Columnas: {df_detalle.columns.tolist()}")
        
        # ===== 2. GUARDAR DATOS_DETALLE =====
        # Si es muy grande, tomar muestra
        if len(df_detalle) > 500000:
            df_detalle_sample = df_detalle.sample(n=500000, random_state=42)
            df_detalle_sample.to_parquet(f'{panel1_dir}/datos_detalle.parquet', index=False)
            print(f"   • datos_detalle.parquet: {len(df_detalle_sample):,} registros (muestra)")
        else:
            df_detalle.to_parquet(f'{panel1_dir}/datos_detalle.parquet', index=False)
            print(f"   • datos_detalle.parquet: {len(df_detalle):,} registros")
        
        # ===== 3. RESÚMENES AGREGADOS =====
        
        # 3.1 Resumen por modelo
        summary = []
        for modelo, col in [('SARIMA', 'sarima_mape'), 
                           ('Prophet', 'prophet_mape'),
                           ('LightGBM', 'lightgbm_mape')]:
            if col in df_results.columns:
                valid = df_results[col].dropna()
                if len(valid) > 0:
                    mae_col = col.replace('mape', 'mae')
                    rmse_col = col.replace('mape', 'rmse')
                    
                    summary.append({
                        'modelo': modelo,
                        'n': len(valid),
                        'mae_media': df_results[mae_col].mean(),
                        'mae_std': df_results[mae_col].std(),
                        'rmse_media': df_results[rmse_col].mean(),
                        'rmse_std': df_results[rmse_col].std(),
                        'mape_media': valid.mean(),
                        'mape_std': valid.std()
                    })
        
        df_summary = pd.DataFrame(summary)
        df_summary.to_parquet(f'{panel1_dir}/resumen_modelos.parquet', index=False)
        print(f"   • resumen_modelos.parquet: {len(df_summary)} modelos")
        
        # 3.2 MAPE por categoría
        if 'categoria_farmaceutica' in df_detalle.columns and 'lightgbm_mape' in df_detalle.columns:
            cat_summary = df_detalle.groupby('categoria_farmaceutica').agg({
                'sarima_mape': ['mean', 'std'],
                'prophet_mape': ['mean', 'std'],
                'lightgbm_mape': ['mean', 'std'],
                'sku_id': 'count'
            }).round(2)
            
            cat_summary.columns = ['_'.join(col).strip() for col in cat_summary.columns.values]
            cat_summary = cat_summary.reset_index().rename(columns={'categoria_farmaceutica': 'categoria'})
            cat_summary.to_parquet(f'{panel1_dir}/mape_por_categoria.parquet', index=False)
            print(f"   • mape_por_categoria.parquet: {len(cat_summary)} categorías")
        
        # 3.3 MAPE por región y tipo
        if 'region' in df_detalle.columns and 'tipo_farmacia' in df_detalle.columns:
            region_summary = df_detalle.groupby(['region', 'tipo_farmacia']).agg({
                'sarima_mape': ['mean', 'std'],
                'prophet_mape': ['mean', 'std'],
                'lightgbm_mape': ['mean', 'std'],
                'sku_id': 'count'
            }).round(2)
            
            region_summary.columns = ['_'.join(col).strip() for col in region_summary.columns.values]
            region_summary = region_summary.reset_index()
            region_summary['clave_region'] = region_summary['region'] + ' - ' + region_summary['tipo_farmacia']
            region_summary.to_parquet(f'{panel1_dir}/mape_por_region.parquet', index=False)
            print(f"   • mape_por_region.parquet: {len(region_summary)} combinaciones")
    
    def create_panel2_kpis(self, df_inventory, df_orders):
        """
        PANEL 2: KPIs operativos e inteligencia de inventario
        """
        print("\n📊 PANEL 2: Creando KPIs de inventario...")
        
        panel2_dir = f'{self.OUTPUT_DIR}/panel2'
        os.makedirs(panel2_dir, exist_ok=True)
        
        if df_inventory.empty:
            print("   ⚠️ No hay datos de inventario")
            return
        
        # ===== 2.1 KPIs globales por semilla =====
        kpis_global = []
        
        for seed in df_inventory['semilla'].unique():
            df_seed = df_inventory[df_inventory['semilla'] == seed]
            
            # Nivel de servicio
            service_level = (df_seed['unidades_vendidas'].sum() / 
                           df_seed['demanda_real'].sum()) if df_seed['demanda_real'].sum() > 0 else 0
            
            # Rotación
            avg_inventory = df_seed['stock_disponible'].mean()
            total_sold = df_seed['unidades_vendidas'].sum()
            turnover = total_sold / avg_inventory if avg_inventory > 0 else 0
            
            # Stockouts
            stockout_rate = (df_seed['stockout_unidades'].sum() / 
                           df_seed['demanda_real'].sum()) if df_seed['demanda_real'].sum() > 0 else 0
            
            kpis_global.append({
                'semilla': seed,
                'nivel_servicio': service_level,
                'rotacion_inventario': turnover,
                'tasa_stockout': stockout_rate,
                'dias_inventario': (avg_inventory / df_seed['demanda_real'].mean()) * 7 if df_seed['demanda_real'].mean() > 0 else 0,
                'total_unidades_vendidas': total_sold,
                'total_unidades_perdidas': df_seed['stockout_unidades'].sum()
            })
        
        df_kpis_global = pd.DataFrame(kpis_global)
        df_kpis_global.to_parquet(f'{panel2_dir}/kpis_global.parquet', index=False)
        print(f"   • kpis_global.parquet: {len(df_kpis_global)} semillas")
        
        # ===== 2.2 KPIs por categoría =====
        if 'categoria' in df_inventory.columns:
            cat_kpis = df_inventory.groupby('categoria').agg({
                'unidades_vendidas': 'sum',
                'demanda_real': 'sum',
                'stockout_unidades': 'sum',
                'stock_disponible': 'mean'
            }).reset_index()
            
            cat_kpis['nivel_servicio'] = cat_kpis['unidades_vendidas'] / cat_kpis['demanda_real']
            cat_kpis['tasa_stockout'] = cat_kpis['stockout_unidades'] / cat_kpis['demanda_real']
            cat_kpis['rotacion'] = cat_kpis['unidades_vendidas'] / cat_kpis['stock_disponible']
            
            cat_kpis.to_parquet(f'{panel2_dir}/kpis_por_categoria.parquet', index=False)
            print(f"   • kpis_por_categoria.parquet: {len(cat_kpis)} categorías")
        
        # ===== 2.3 KPIs por región y tipo =====
        if 'region' in df_inventory.columns and 'tipo_farmacia' in df_inventory.columns:
            region_kpis = df_inventory.groupby(['region', 'tipo_farmacia']).agg({
                'unidades_vendidas': 'sum',
                'demanda_real': 'sum',
                'stockout_unidades': 'sum',
                'stock_disponible': 'mean'
            }).reset_index()
            
            region_kpis['nivel_servicio'] = region_kpis['unidades_vendidas'] / region_kpis['demanda_real']
            region_kpis['tasa_stockout'] = region_kpis['stockout_unidades'] / region_kpis['demanda_real']
            region_kpis['rotacion'] = region_kpis['unidades_vendidas'] / region_kpis['stock_disponible']
            region_kpis['clave_region'] = region_kpis['region'] + ' - ' + region_kpis['tipo_farmacia']
            
            region_kpis.to_parquet(f'{panel2_dir}/kpis_por_region.parquet', index=False)
            print(f"   • kpis_por_region.parquet: {len(region_kpis)} combinaciones")
        
        # ===== 2.4 Órdenes =====
        if not df_orders.empty:
            df_orders.to_parquet(f'{panel2_dir}/ordenes_detalle.parquet', index=False)
            print(f"   • ordenes_detalle.parquet: {len(df_orders)} registros")
    
    def create_panel3_analisis(self, df_results):
        """
        PANEL 3: Análisis de errores y recomendaciones
        """
        print("\n📊 PANEL 3: Creando análisis de errores...")
        
        panel3_dir = f'{self.OUTPUT_DIR}/panel3'
        os.makedirs(panel3_dir, exist_ok=True)
        
        if df_results.empty:
            print("   ⚠️ No hay datos de resultados")
            return
        
        # ===== 3.1 Análisis de errores por modelo =====
        error_analysis = []
        
        for modelo, col in [('SARIMA', 'sarima_mape'), 
                           ('Prophet', 'prophet_mape'),
                           ('LightGBM', 'lightgbm_mape')]:
            if col in df_results.columns:
                valid = df_results[col].dropna()
                if len(valid) > 0:
                    error_analysis.append({
                        'modelo': modelo,
                        'mape_p25': valid.quantile(0.25),
                        'mape_p50': valid.median(),
                        'mape_p75': valid.quantile(0.75),
                        'mape_min': valid.min(),
                        'mape_max': valid.max(),
                        'sku_count': len(valid)
                    })
        
        df_error_analysis = pd.DataFrame(error_analysis)
        df_error_analysis.to_parquet(f'{panel3_dir}/error_analysis.parquet', index=False)
        print(f"   • error_analysis.parquet: {len(df_error_analysis)} modelos")
        
        # ===== 3.2 Mejores y peores SKUs =====
        if 'lightgbm_mape' in df_results.columns:
            best_skus = df_results.nsmallest(20, 'lightgbm_mape')[
                ['sku_farm_id', 'categoria', 'region', 'tipo_farmacia', 
                 'sarima_mape', 'prophet_mape', 'lightgbm_mape']
            ]
            best_skus.to_parquet(f'{panel3_dir}/best_skus.parquet', index=False)
            
            worst_skus = df_results.nlargest(20, 'lightgbm_mape')[
                ['sku_farm_id', 'categoria', 'region', 'tipo_farmacia', 
                 'sarima_mape', 'prophet_mape', 'lightgbm_mape']
            ]
            worst_skus.to_parquet(f'{panel3_dir}/worst_skus.parquet', index=False)
            print(f"   • best/worst SKUs guardados")
        
        # ===== 3.3 Correlación =====
        corr_matrix = df_results[['sarima_mape', 'prophet_mape', 'lightgbm_mape']].corr()
        corr_matrix.to_parquet(f'{panel3_dir}/correlacion_modelos.parquet')
        print(f"   • correlacion_modelos.parquet")

    def create_sqlite_database(self):
        """
        Crea base de datos SQLite optimizada para Power BI
        CON RELACIONES ENTIDAD-RELACIÓN
        """
        print("\n💾 Creando base de datos SQLite con relaciones...")
        
        db_path = f'{self.OUTPUT_DIR}/powerbi_optimized.db'
        
        if os.path.exists(db_path):
            os.remove(db_path)
        
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA cache_size = -2000000")
        
        cursor = conn.cursor()
        
        # ===== 1. CREAR TABLAS EN ORDEN CORRECTO =====
        
        # Primero las dimensiones
        if os.path.exists(f'{self.OUTPUT_DIR}/dim_fechas.parquet'):
            df_fechas = pd.read_parquet(f'{self.OUTPUT_DIR}/dim_fechas.parquet')
            df_fechas.to_sql('dim_fechas', conn, if_exists='replace', index=False)
            cursor.execute("CREATE INDEX idx_fechas_fecha ON dim_fechas(fecha);")
        
        if os.path.exists(f'{self.OUTPUT_DIR}/dim_productos.parquet'):
            df_productos = pd.read_parquet(f'{self.OUTPUT_DIR}/dim_productos.parquet')
            df_productos.to_sql('dim_productos', conn, if_exists='replace', index=False)
            cursor.execute("CREATE INDEX idx_productos_sku ON dim_productos(sku_id);")
        
        if os.path.exists(f'{self.OUTPUT_DIR}/dim_tiendas.parquet'):
            df_tiendas = pd.read_parquet(f'{self.OUTPUT_DIR}/dim_tiendas.parquet')
            df_tiendas.to_sql('dim_tiendas', conn, if_exists='replace', index=False)
            cursor.execute("CREATE INDEX idx_tiendas_id ON dim_tiendas(tienda_id);")
        
        if os.path.exists(f'{self.OUTPUT_DIR}/dim_modelos.parquet'):
            df_modelos = pd.read_parquet(f'{self.OUTPUT_DIR}/dim_modelos.parquet')
            df_modelos.to_sql('dim_modelos', conn, if_exists='replace', index=False)
        
        # Luego la tabla de hechos (datos_detalle)
        if os.path.exists(f'{self.OUTPUT_DIR}/panel1/datos_detalle.parquet'):
            df_detalle = pd.read_parquet(f'{self.OUTPUT_DIR}/panel1/datos_detalle.parquet')
            df_detalle.to_sql('datos_detalle', conn, if_exists='replace', index=False)
            
            # Crear índices para las claves foráneas
            cursor.execute("CREATE INDEX idx_detalle_fecha ON datos_detalle(fecha);")
            cursor.execute("CREATE INDEX idx_detalle_farmacia ON datos_detalle(farmacia_id);")
            cursor.execute("CREATE INDEX idx_detalle_sku ON datos_detalle(sku_id);")
            cursor.execute("CREATE INDEX idx_detalle_semilla ON datos_detalle(semilla);")
        
        # Tablas de resumen de paneles
        for panel in ['panel1', 'panel2', 'panel3']:
            panel_dir = f'{self.OUTPUT_DIR}/{panel}'
            if os.path.exists(panel_dir):
                for parquet_file in glob.glob(f'{panel_dir}/*.parquet'):
                    if 'datos_detalle' not in parquet_file:  # Ya la cargamos
                        table_name = os.path.splitext(os.path.basename(parquet_file))[0]
                        df = pd.read_parquet(parquet_file)
                        df.to_sql(table_name, conn, if_exists='replace', index=False)
                        print(f"   • Tabla {table_name} cargada: {len(df)} registros")
        
        # ===== 2. VERIFICACIÓN =====
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\n   📊 Tablas en BD con relaciones:")
        total_rows = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
            count = cursor.fetchone()[0]
            print(f"      • {table[0]}: {count:,} registros")
            total_rows += count
        
        db_size = os.path.getsize(db_path) / (1024**3)
        print(f"\n   ✅ BD creada: {db_size:.2f} GB - {total_rows:,} registros totales")
        
        conn.close()
    
    def run(self):
        """
        Ejecuta todo el proceso de unificación
        """
        print("\n" + "=" * 80)
        print("🚀 UNIFICACIÓN FINAL PARA POWER BI - VERSIÓN DEFINITIVA")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. Cargar resultados de las 3 ejecuciones
        df_results = self.load_all_executions()
        
        # 2. Cargar datos de demanda real (con fechas)
        df_demand = self.load_demand_data()
        
        # 3. Cargar datos de inventario
        df_inventory, df_orders = self.load_inventory_data()
        
        # 4. Crear dimensiones (basadas en demanda)
        self.create_dimensions(df_demand)
        
        # 5. Crear paneles
        self.create_panel1_comparativa(df_results, df_demand)
        self.create_panel2_kpis(df_inventory, df_orders)
        self.create_panel3_analisis(df_results)
        
        # 6. Crear base SQLite con relaciones
        self.create_sqlite_database()
        
        # 7. Limpiar temporales
        if os.path.exists(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {elapsed/60:.1f} minutos")
        
        print("\n" + "=" * 80)
        print("✅ UNIFICACIÓN COMPLETADA")
        print("=" * 80)
        print(f"\n📁 Archivos generados en: {self.OUTPUT_DIR}")


def main():
    print("=" * 80)
    print("🎯 UNIFICADOR POWER BI - VERSIÓN DEFINITIVA")
    print("   Con relaciones entidad-relación")
    print("=" * 80)
    
    unifier = PowerBIFinalUnifier(num_workers=4)
    unifier.run()


if __name__ == "__main__":
    main()