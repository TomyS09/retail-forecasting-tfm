# src/inventory_models.py
"""
Modelos avanzados de gestión de inventario para farmacias
Incluye políticas (s,Q), (R,S), Newsvendor, y basadas en ML
"""

import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum

class InventoryPolicy(Enum):
    """Políticas de inventario"""
    BASE_STOCK = "base_stock"       # Nivel objetivo constante
    REORDER_POINT = "reorder_point" # (s,Q) - Punto de reorden, cantidad fija
    PERIODIC_REVIEW = "periodic"    # Revisión periódica (R,S)
    NEWSVENDOR = "newsvendor"       # Para productos perecederos
    ML_OPTIMIZED = "ml_optimized"   # Optimizado por machine learning

class InventoryItem:
    """Item de inventario con características específicas"""
    
    def __init__(self, sku_id, categoria, costo_unitario, lead_time_dias,
                 holding_cost_rate=0.25, stockout_cost_multiplier=2.0,
                 service_level_target=0.95):
        """
        Parámetros:
        - holding_cost_rate: Costo anual de mantener inventario (25% típico)
        - stockout_cost_multiplier: Multiplicador del margen perdido por stockout
        - service_level_target: Nivel de servicio objetivo (95%)
        """
        self.sku_id = sku_id
        self.categoria = categoria
        self.costo_unitario = costo_unitario
        self.lead_time_dias = lead_time_dias
        self.lead_time_semanas = lead_time_dias / 7
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_multiplier = stockout_cost_multiplier
        self.service_level_target = service_level_target
        
        # Calcular costo semanal de mantener
        self.holding_cost_semanal = costo_unitario * holding_cost_rate / 52
        
        # Historial de demanda para cálculos
        self.demand_history = []
        self.forecast_history = {}
        
    def add_demand(self, demanda_real, fecha):
        """Agrega demanda real al historial"""
        self.demand_history.append({
            'fecha': fecha,
            'demanda': demanda_real
        })
        
    def add_forecast(self, modelo, prediccion, fecha):
        """Agrega predicción de un modelo"""
        if modelo not in self.forecast_history:
            self.forecast_history[modelo] = []
        self.forecast_history[modelo].append({
            'fecha': fecha,
            'prediccion': prediccion
        })
    
    def calculate_statistics(self, window_weeks=26):
        """Calcula estadísticas de demanda reciente"""
        if len(self.demand_history) < 4:
            return None
            
        # Usar últimas N semanas
        recent_demands = [d['demanda'] for d in self.demand_history[-window_weeks:]]
        
        if not recent_demands:
            return None
            
        return {
            'mean': np.mean(recent_demands),
            'std': np.std(recent_demands),
            'cv': np.std(recent_demands) / np.mean(recent_demands) if np.mean(recent_demands) > 0 else 0,
            'max': np.max(recent_demands),
            'min': np.min(recent_demands),
            'zero_rate': sum(1 for d in recent_demands if d == 0) / len(recent_demands)
        }

class InventoryOptimizer:
    """Optimizador de políticas de inventario"""
    
    def __init__(self):
        self.policies = {}
        
    def calculate_eoq(self, item, demanda_anual, costo_orden):
        """
        Fórmula EOQ (Economic Order Quantity) clásica
        Q* = √(2DS/H)
        donde: D = demanda anual, S = costo por orden, H = costo de mantener por unidad por año
        """
        if item.costo_unitario <= 0:
            return max(1, int(demanda_anual / 52))  # 1 semana de demanda
        
        H = item.costo_unitario * item.holding_cost_rate
        if H <= 0:
            return max(1, int(demanda_anual / 26))  # 2 semanas de demanda
            
        eoq = np.sqrt((2 * demanda_anual * costo_orden) / H)
        return max(1, int(eoq))
    
    def calculate_safety_stock(self, item, stats_dict, service_level=0.95):
        """
        Calcula stock de seguridad basado en variabilidad de demanda y lead time
        SS = z * σ * √L
        donde: z = valor Z para nivel de servicio, σ = desviación estándar demanda, L = lead time
        """
        if stats_dict is None or stats_dict['std'] == 0:
            return 0
            
        # Valor Z para nivel de servicio (CORREGIDO)
        z_value = stats.norm.ppf(service_level)
        
        # Stock de seguridad clásico
        safety_stock = z_value * stats_dict['std'] * np.sqrt(item.lead_time_semanas)
        
        # Ajustar por tipo de producto
        if stats_dict['cv'] > 1.0:  # Alta variabilidad
            safety_stock *= 1.5
        elif stats_dict['cv'] < 0.3:  # Baja variabilidad
            safety_stock *= 0.7
            
        return max(0, int(safety_stock))
    
    def calculate_newsvendor(self, item, stats_dict, costo_sobre, costo_falta):
        """
        Modelo Newsvendor para productos perecederos
        Nivel óptimo = distribución inversa de (cu / (cu + co))
        donde: cu = costo de falta, co = costo de sobre
        """
        if stats_dict is None:
            return 0
            
        critical_ratio = costo_falta / (costo_falta + costo_sobre)
        
        # Asumir distribución normal de demanda
        if stats_dict['std'] > 0:
            optimal_level = stats_dict['mean'] + stats_dict['std'] * stats.norm.ppf(critical_ratio)
        else:
            optimal_level = stats_dict['mean']
            
        return max(0, int(optimal_level))
    
    def recommend_policy(self, item, stats_dict, costo_orden=50):
        """
        Recomienda política óptima basada en características del producto
        """
        if stats_dict is None:
            return InventoryPolicy.REORDER_POINT, {'s': 10, 'Q': 20}
        
        # Clasificar producto según características
        if stats_dict['zero_rate'] > 0.3:
            # Producto intermitente
            if stats_dict['cv'] > 1.5:
                policy = InventoryPolicy.PERIODIC_REVIEW
                review_period = 4  # Revisar cada 4 semanas
                order_up_to = int(stats_dict['mean'] * (review_period + item.lead_time_semanas) * 1.5)
                params = {'R': review_period, 'S': order_up_to}
            else:
                policy = InventoryPolicy.REORDER_POINT
                demanda_anual = stats_dict['mean'] * 52
                Q = self.calculate_eoq(item, demanda_anual, costo_orden)
                safety_stock = self.calculate_safety_stock(item, stats_dict)
                s = int(stats_dict['mean'] * item.lead_time_semanas + safety_stock)
                params = {'s': max(1, s), 'Q': max(1, Q)}
                
        elif stats_dict['cv'] < 0.3:
            # Producto de demanda estable (crónicos)
            policy = InventoryPolicy.BASE_STOCK
            safety_stock = self.calculate_safety_stock(item, stats_dict)
            S = int(stats_dict['mean'] * (1 + item.lead_time_semanas) + safety_stock)
            params = {'S': max(1, S)}
            
        elif "Perecedero" in item.categoria or "Antibiótico" in item.categoria:
            # Productos perecederos (NewsVendor)
            policy = InventoryPolicy.NEWSVENDOR
            costo_sobre = item.costo_unitario * 0.3  # 30% pérdida por sobrestock
            costo_falta = item.costo_unitario * item.stockout_cost_multiplier
            optimal_level = self.calculate_newsvendor(item, stats_dict, costo_sobre, costo_falta)
            params = {'optimal_level': int(optimal_level)}
            
        else:
            # Producto estándar
            policy = InventoryPolicy.REORDER_POINT
            demanda_anual = stats_dict['mean'] * 52
            Q = self.calculate_eoq(item, demanda_anual, costo_orden)
            safety_stock = self.calculate_safety_stock(item, stats_dict)
            s = int(stats_dict['mean'] * item.lead_time_semanas + safety_stock)
            params = {'s': max(1, s), 'Q': max(1, Q)}
        
        return policy, params
    
    def generate_order_recommendation(self, item, current_stock, modelo_forecast=None):
        """
        Genera recomendación de orden basada en política y estado actual
        """
        stats_dict = item.calculate_statistics()
        if stats_dict is None:
            return {'ordenar': 0, 'razon': 'datos_insuficientes'}
        
        policy, params = self.recommend_policy(item, stats_dict)
        
        if policy == InventoryPolicy.REORDER_POINT:
            s, Q = params['s'], params['Q']
            if current_stock <= s:
                return {
                    'ordenar': Q,
                    'razon': f'stock ({current_stock}) ≤ punto_reorden ({s})',
                    'politica': 'reorder_point',
                    'punto_reorden': s,
                    'cantidad_orden': Q
                }
                
        elif policy == InventoryPolicy.BASE_STOCK:
            S = params['S']
            if current_stock < S:
                orden = S - current_stock
                return {
                    'ordenar': orden,
                    'razon': f'stock ({current_stock}) < nivel_objetivo ({S})',
                    'politica': 'base_stock',
                    'nivel_objetivo': S,
                    'deficit': orden
                }
                
        elif policy == InventoryPolicy.PERIODIC_REVIEW:
            R, S = params['R'], params['S']
            # En simulación simple, ordenar cada R periodos
            # Implementar lógica más compleja en simulación real
            orden = max(0, S - current_stock)
            return {
                'ordenar': orden,
                'razon': f'revision_periodica (cada {R} semanas)',
                'politica': 'periodic_review',
                'periodo_revision': R,
                'nivel_maximo': S
            }
            
        elif policy == InventoryPolicy.NEWSVENDOR:
            optimal = params['optimal_level']
            if current_stock < optimal:
                orden = optimal - current_stock
                return {
                    'ordenar': orden,
                    'razon': f'optimizacion_newsvendor (nivel_optimo: {optimal})',
                    'politica': 'newsvendor',
                    'nivel_optimo': optimal
                }
        
        return {'ordenar': 0, 'razon': 'stock_suficiente'}