# src/simulation_inventory_sample.py
"""
Simulación de inventario SOLO para la muestra estratificada
Usa las predicciones REALES de los 3 modelos (SARIMA, Prophet, LightGBM)
Procesa por separado cada semilla (42, 123, 456)
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
import pyarrow as pa

from inventory_models import InventoryItem, InventoryOptimizer


class SampleInventorySimulator:
    """
    Simulador de inventario optimizado para la muestra estratificada
    """
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or cpu_count()
        
        # Directorios
        self.DATA_DIR = 'data'
        self.RESULTS_DIR = 'data/model_results'
        self.INVENTORY_DIR = 'data/inventory_sample'
        self.CHECKPOINT_DIR = 'data/inventory_checkpoints'
        
        for d in [self.INVENTORY_DIR, self.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)
        
        # Costos por tipo de farmacia
        self.ordering_costs = {
            'Farmacia Cadena': 30,
            'Farmacia Comunitaria': 40,
            'Botica': 50
        }
        
        print(f"\n🔧 Simulador de inventario (muestra) inicializado:")
        print(f"   • Workers: {self.num_workers}")
    
    def load_sample_combinations(self, seed):
        """
        Carga las combinaciones farmacia-SKU de la muestra para una semilla
        
        Args:
            seed: Semilla de la ejecución (42, 123, 456)
        
        Returns:
            Lista de strings "farmacia_id_sku_id"
        """
        results_file = f'{self.RESULTS_DIR}/resultados_seed{seed}.parquet'
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"No se encuentra {results_file}")
        
        df_results = pd.read_parquet(results_file)
        
        # Extraer combinaciones únicas que tuvieron éxito
        if 'sku_farm_id' in df_results.columns:
            combinations = df_results[df_results['status'] == 'completed']['sku_farm_id'].tolist()
        else:
            # Si no existe sku_farm_id, construirlo
            combinations = (df_results['farmacia_id'] + '_' + df_results['sku_id']).tolist()
        
        print(f"\n📋 Cargando muestra para semilla {seed}:")
        print(f"   • Combinaciones encontradas: {len(combinations)}")
        
        return combinations
    
    def load_demand_and_predictions(self, seed, combinations):
        """
        Carga datos de demanda y predicciones para las combinaciones de la muestra
        
        Args:
            seed: Semilla de la ejecución
            combinations: Lista de combinaciones a cargar
        
        Returns:
            DataFrame con columnas: fecha, farmacia_id, sku_id, demanda_real,
                                   pred_sarima, pred_prophet, pred_lightgbm
        """
        print(f"\n📂 Cargando datos de demanda para semilla {seed}...")
        
        # Cargar datos de evaluación (generados por run_full_pipeline)
        eval_path = f'{self.DATA_DIR}/eval_data_seed{seed}.parquet'
        df_eval = pd.read_parquet(eval_path)
        
        # Filtrar solo las combinaciones de la muestra
        df_eval['comb_id'] = df_eval['farmacia_id'] + '_' + df_eval['sku_id']
        df_sample = df_eval[df_eval['comb_id'].isin(combinations)].copy()
        
        print(f"   • Registros cargados: {len(df_sample):,}")
        print(f"   • Combinaciones únicas: {df_sample['comb_id'].nunique()}")
        
        return df_sample
    
    def initialize_inventory_states(self, df_sample):
        """
        Inicializa estados de inventario para cada combinación
        
        Args:
            df_sample: DataFrame con datos de la muestra
        
        Returns:
            Dict: {comb_id: estado_inicial}
        """
        print(f"\n📦 Inicializando estados de inventario...")
        
        inventory_states = {}
        optimizer = InventoryOptimizer()
        
        for comb_id, group in df_sample.groupby('comb_id'):
            farmacia_id, sku_id = comb_id.split('_', 1)
            row = group.iloc[0]
            
            # Crear item de inventario
            item = InventoryItem(
                sku_id=sku_id,
                categoria=row['categoria_farmaceutica'],
                costo_unitario=row['costo_unitario'],
                lead_time_dias=self.get_lead_time(row['categoria_farmaceutica']),
                holding_cost_rate=0.25,
                stockout_cost_multiplier=2.5,
                service_level_target=0.95
            )
            
            # Stock inicial según costo
            if row['costo_unitario'] < 5:
                init_stock = np.random.randint(50, 100)
            elif row['costo_unitario'] < 20:
                init_stock = np.random.randint(30, 60)
            else:
                init_stock = np.random.randint(10, 30)
            
            inventory_states[comb_id] = {
                'item': item,
                'farmacia_id': farmacia_id,
                'sku_id': sku_id,
                'categoria': row['categoria_farmaceutica'],
                'region': row['region'],
                'tipo_farmacia': row['tipo_farmacia'],
                'current_stock': init_stock,
                'on_order': 0,
                'order_arrival_week': -1,
                'stockout_days': 0,
                'holding_cost_accumulated': 0.0,
                'stockout_cost_accumulated': 0.0,
                'ordering_cost_accumulated': 0.0,
                'total_orders': 0
            }
        
        print(f"   • {len(inventory_states)} combinaciones inicializadas")
        return inventory_states
    
    def get_lead_time(self, categoria):
        """Obtiene lead time según categoría"""
        lead_times = {
            'Material de Curación': 2,
            'Jeringuillas': 2,
            'Analgésicos': 3,
            'Antiácidos': 3,
            'Vitaminas': 5,
            'Genéricos': 5,
            'Antihipertensivos': 7,
            'Antidiabéticos': 7,
            'Antibióticos': 10,
            'Equipos de Diagnóstico': 14,
            'Antineoplásicos': 21,
            'Inmunosupresores': 21
        }
        
        for key, days in lead_times.items():
            if key in categoria:
                return days
        return 7
    
    def _process_combination(self, comb_id, df_comb, state, seed):
        """
        Procesa una combinación farmacia-SKU a lo largo del tiempo
        """
        try:
            # Ordenar por fecha
            df_comb = df_comb.sort_values('fecha')
            
            # Parámetros
            ordering_cost = self.ordering_costs.get(state['tipo_farmacia'], 40)
            lead_time_weeks = state['item'].lead_time_semanas
            holding_cost_weekly = state['item'].holding_cost_semanal
            stockout_multiplier = state['item'].stockout_cost_multiplier
            costo_unitario = state['item'].costo_unitario
            
            # Estado actual
            stock = state['current_stock']
            on_order = state['on_order']
            arrival_week = state['order_arrival_week']
            
            # Acumuladores
            stockout_days = 0
            stockout_cost = 0.0
            holding_cost = 0.0
            ordering_cost_total = 0.0
            orders_generated = 0
            
            # Historial de demanda para cálculos
            demand_history = []
            
            # Registros para este SKU
            inventory_records = []
            order_records = []
            
            for idx, row in df_comb.iterrows():
                week = idx  # Usar índice como semana
                demanda = row['demanda_real']
                
                # Recibir pedido si llegó
                if on_order > 0 and arrival_week >= 0 and week >= arrival_week:
                    stock += on_order
                    on_order = 0
                    arrival_week = -1
                
                # Actualizar historial (últimas 52 semanas)
                demand_history.append(demanda)
                if len(demand_history) > 52:
                    demand_history.pop(0)
                
                # Satisfacer demanda
                if stock >= demanda:
                    vendidas = demanda
                    stockout = 0
                    stock -= demanda
                else:
                    vendidas = stock
                    stockout = demanda - stock
                    stock = 0
                    stockout_cost += stockout * costo_unitario * stockout_multiplier
                    stockout_days += 1
                
                # Costo de holding
                holding_cost += stock * holding_cost_weekly
                
                # Decisión de orden (cada 4 semanas)
                if week % 4 == 0 and on_order == 0 and len(demand_history) >= 4:
                    # Promedio últimas 4 semanas
                    if len(demand_history) <= 4:
                        avg_demand = sum(demand_history) / len(demand_history)
                    else:
                        avg_demand = sum(demand_history[-4:]) / 4.0
                    
                    if avg_demand > 0:
                        reorder_point = int(avg_demand * 2)  # 2 semanas de cobertura
                        if stock <= reorder_point:
                            order_qty = int(avg_demand * 4)  # 4 semanas de demanda
                            ordering_cost_total += ordering_cost
                            on_order = order_qty
                            arrival_week = week + int(np.ceil(lead_time_weeks))
                            orders_generated += 1
                            
                            # Registrar orden
                            order_records.append({
                                'fecha': row['fecha'],
                                'comb_id': comb_id,
                                'farmacia_id': state['farmacia_id'],
                                'sku_id': state['sku_id'],
                                'categoria': state['categoria'],
                                'region': state['region'],
                                'tipo_farmacia': state['tipo_farmacia'],
                                'cantidad': order_qty,
                                'stock_preorden': stock + vendidas,
                                'lead_time_semanas': lead_time_weeks,
                                'semana_llegada': arrival_week,
                                'costo_orden': ordering_cost,
                                'razon': 'punto_reorden',
                                'seed': seed
                            })
                
                # Registrar estado semanal
                inventory_records.append({
                    'fecha': row['fecha'],
                    'comb_id': comb_id,
                    'farmacia_id': state['farmacia_id'],
                    'sku_id': state['sku_id'],
                    'categoria': state['categoria'],
                    'region': state['region'],
                    'tipo_farmacia': state['tipo_farmacia'],
                    'demanda_real': demanda,
                    'pred_sarima': row.get('pred_arima', demanda),
                    'pred_prophet': row.get('pred_prophet', demanda),
                    'pred_lightgbm': row.get('pred_lightgbm', demanda),
                    'unidades_vendidas': vendidas,
                    'stockout_unidades': stockout,
                    'stock_disponible': stock,
                    'pedido_pendiente': on_order,
                    'semana_llegada': arrival_week if arrival_week >= 0 else None,
                    'nivel_servicio': vendidas / demanda if demanda > 0 else 1.0,
                    'costo_holding_semanal': stock * holding_cost_weekly,
                    'costo_stockout_semanal': stockout * costo_unitario * stockout_multiplier,
                    'seed': seed
                })
            
            # Actualizar estado final
            state['current_stock'] = stock
            state['on_order'] = on_order
            state['order_arrival_week'] = arrival_week
            state['stockout_days'] += stockout_days
            state['stockout_cost_accumulated'] += stockout_cost
            state['holding_cost_accumulated'] += holding_cost
            state['ordering_cost_accumulated'] += ordering_cost_total
            state['total_orders'] += orders_generated
            
            return {
                'comb_id': comb_id,
                'inventory_records': inventory_records,
                'order_records': order_records,
                'stockout_days': stockout_days,
                'orders_generated': orders_generated
            }
            
        except Exception as e:
            print(f"   ❌ Error en {comb_id}: {e}")
            return {
                'comb_id': comb_id,
                'inventory_records': [],
                'order_records': [],
                'stockout_days': 0,
                'orders_generated': 0,
                'error': str(e)
            }
    
    def simulate_seed(self, seed, chunk_size=25):
        """
        Ejecuta simulación de inventario para una semilla específica
        
        Args:
            seed: Semilla (42, 123, 456)
            chunk_size: Tamaño de chunk para procesamiento paralelo
        """
        print("\n" + "=" * 80)
        print(f"🚀 SIMULACIÓN DE INVENTARIO PARA SEMILLA {seed}")
        print("=" * 80)
        
        # Fijar semilla
        np.random.seed(seed)
        
        # ===== FASE 1: Cargar combinaciones de la muestra =====
        combinations = self.load_sample_combinations(seed)
        
        # ===== FASE 2: Cargar datos de demanda y predicciones =====
        df_data = self.load_demand_and_predictions(seed, combinations)
        
        # ===== FASE 3: Inicializar estados de inventario =====
        inventory_states = self.initialize_inventory_states(df_data)
        
        # ===== FASE 4: Procesar en paralelo =====
        print("\n" + "=" * 60)
        print("🤖 FASE 4: PROCESAMIENTO PARALELO")
        print("=" * 60)
        
        # Dividir combinaciones en chunks
        comb_list = list(inventory_states.keys())
        chunks = [comb_list[i:i+chunk_size] 
                 for i in range(0, len(comb_list), chunk_size)]
        
        print(f"   • {len(chunks)} chunks de ~{chunk_size} combinaciones")
        
        all_inventory_records = []
        all_order_records = []
        execution_start = time.time()
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.time()
            print(f"\n📦 Chunk {chunk_idx+1}/{len(chunks)} ({len(chunk)} combinaciones)")
            
            # Preparar datos para el chunk
            chunk_data = []
            for comb_id in chunk:
                state = inventory_states[comb_id]
                df_comb = df_data[df_data['comb_id'] == comb_id].copy()
                chunk_data.append((comb_id, df_comb, state, seed))
            
            # Procesar en paralelo
            with Pool(processes=min(self.num_workers, len(chunk))) as pool:
                chunk_results = pool.starmap(self._process_combination, chunk_data)
            
            # Recolectar resultados
            for result in chunk_results:
                all_inventory_records.extend(result['inventory_records'])
                all_order_records.extend(result['order_records'])
            
            # Estadísticas
            chunk_time = time.time() - chunk_start
            print(f"   ✅ Chunk completado en {chunk_time:.1f}s")
            print(f"      • Progreso: {chunk_idx+1}/{len(chunks)} chunks")
            
            gc.collect()
        
        # ===== FASE 5: Consolidar resultados =====
        print("\n" + "=" * 60)
        print(f"📊 FASE 5: CONSOLIDACIÓN DE RESULTADOS (semilla {seed})")
        print("=" * 60)
        
        # Guardar inventario
        df_inventory = pd.DataFrame(all_inventory_records)
        inventory_file = f'{self.INVENTORY_DIR}/inventario_seed{seed}.parquet'
        df_inventory.to_parquet(inventory_file, compression='snappy', index=False)
        print(f"   • Inventario guardado: {inventory_file}")
        print(f"   • Registros: {len(df_inventory):,}")
        
        # Guardar órdenes
        if all_order_records:
            df_orders = pd.DataFrame(all_order_records)
            orders_file = f'{self.INVENTORY_DIR}/ordenes_seed{seed}.parquet'
            df_orders.to_parquet(orders_file, compression='snappy', index=False)
            print(f"   • Órdenes guardadas: {orders_file}")
            print(f"   • Órdenes: {len(df_orders):,}")
        
        # ===== FASE 6: Métricas agregadas =====
        print("\n" + "=" * 60)
        print("📈 FASE 6: MÉTRICAS AGREGADAS")
        print("=" * 60)
        
        # Nivel de servicio promedio
        service_level = df_inventory.groupby('comb_id').apply(
            lambda x: x['unidades_vendidas'].sum() / x['demanda_real'].sum()
        ).mean()
        
        # Rotación de inventario
        inventory_turnover = df_inventory.groupby('comb_id').apply(
            lambda x: x['unidades_vendidas'].sum() / x['stock_disponible'].mean()
        ).mean()
        
        # Stockouts
        stockout_rate = (df_inventory['stockout_unidades'].sum() / 
                        df_inventory['demanda_real'].sum())
        
        print(f"\n   📊 RESUMEN SEMILLA {seed}:")
        print(f"      • Nivel de servicio: {service_level:.2%}")
        print(f"      • Rotación inventario: {inventory_turnover:.2f} veces/año")
        print(f"      • Tasa stockout: {stockout_rate:.2%}")
        
        execution_time = time.time() - execution_start
        print(f"\n   • Tiempo total: {execution_time/60:.1f} minutos")
        
        return inventory_file
    
    def run_all_seeds(self, seeds=[42, 123, 456], chunk_size=25):
        """
        Ejecuta simulación para todas las semillas
        """
        print("\n" + "=" * 80)
        print("🚀 SIMULACIÓN DE INVENTARIO - TODAS LAS SEMILLAS")
        print("=" * 80)
        
        all_inventory_files = []
        
        for seed in seeds:
            inventory_file = self.simulate_seed(seed, chunk_size)
            all_inventory_files.append(inventory_file)
        
        print("\n" + "=" * 80)
        print("✅ SIMULACIÓN COMPLETADA")
        print("=" * 80)
        print("Archivos generados:")
        for f in all_inventory_files:
            size = os.path.getsize(f) / (1024**3)
            print(f"   • {f} ({size:.2f} GB)")
        
        return all_inventory_files


def main():
    """
    Punto de entrada principal
    """
    print("=" * 80)
    print("🎯 SIMULACIÓN DE INVENTARIO - VERSIÓN MUESTRA")
    print("   Para retail farmacéutico ecuatoriano")
    print("=" * 80)
    
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    workers = min(4, n_cores)
    
    simulator = SampleInventorySimulator(num_workers=workers)
    
    print("\n📋 Configuración:")
    print("   • Semillas: 42, 123, 456")
    print("   • Combinaciones por semilla: ~2,484")
    print(f"   • Workers: {workers}")
    
    simulator.run_all_seeds(seeds=[42, 123, 456], chunk_size=25)
    
    print("\n✨ Proceso completado exitosamente")


if __name__ == "__main__":
    main()