# src/models_sarima.py
"""
Modelo SARIMA para predicción de demanda farmacéutica
Referencias:
- Box, G.E.P., et al. (2016). Time Series Analysis: Forecasting and Control
- Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os

class SARIMAModel:
    """
    Modelo SARIMA con auto-selección de parámetros
    """
    
    def __init__(self, seasonal_period=52, max_p=3, max_d=1, max_q=3, 
                 max_P=2, max_D=1, max_Q=2, m=52):
        """
        Parámetros:
            seasonal_period: período estacional (52 semanas = 1 año)
            max_p, max_d, max_q: órdenes máximos ARIMA no estacional
            max_P, max_D, max_Q: órdenes máximos SARIMA estacional
            m: número de períodos por temporada
        """
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.m = m
        self.model = None
        self.model_fit = None
        self.sku_id = None
        
    def fit(self, series, sku_id=None):
        """
        Ajusta modelo SARIMA con auto-selección de parámetros
        Referencia: Hyndman & Khandakar (2008) - "Automatic time series forecasting"
        """
        self.sku_id = sku_id
        
        # Asegurar que la serie es estacionaria (diferenciación automática)
        try:
            # Usar pmdarima para selección automática
            self.model = pm.auto_arima(
                series,
                seasonal=True,
                m=self.m,
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                max_P=self.max_P,
                max_D=self.max_D,
                max_Q=self.max_Q,
                start_p=0,
                start_q=0,
                start_P=0,
                start_Q=0,
                information_criterion='aic',  # AIC para selección
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,  # Búsqueda paso a paso (más rápida)
                n_fits=50
            )
            
            print(f"      ✓ SARIMA{self.model.order}x{self.model.seasonal_order} - AIC: {self.model.aic():.2f}")
            
        except Exception as e:
            print(f"      ⚠️ Error en auto_arima: {e}. Usando SARIMA(1,0,1)(1,0,1,52)")
            # Fallback a modelo simple
            self.model = pm.auto_arima(
                series,
                seasonal=True,
                m=self.m,
                max_p=1, max_d=0, max_q=1,
                max_P=1, max_D=0, max_Q=1,
                information_criterion='aic',
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
        
        return self
    
    def predict(self, n_periods=1, return_conf_int=False):
        """
        Predice n períodos hacia adelante
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")
        
        forecast, conf_int = self.model.predict(n_periods=n_periods, return_conf_int=True)
        
        if return_conf_int:
            return forecast, conf_int
        return forecast
    
    def save(self, path):
        """Guarda el modelo"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """Carga un modelo guardado"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def train_sarima_for_sku(sku_data, sku_id, output_dir='models/sarima'):
    """
    Entrena modelo SARIMA para un SKU específico
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ordenar por fecha y extraer serie
    sku_data = sku_data.sort_values('fecha')
    series = sku_data['demanda_real'].values
    
    # Verificar longitud suficiente
    if len(series) < 78:  # Mínimo 1.5 años
        print(f"      ⚠️ SKU {sku_id}: datos insuficientes ({len(series)} semanas)")
        return None
    
    try:
        # Entrenar modelo
        model = SARIMAModel(seasonal_period=52)
        model.fit(series, sku_id=sku_id)
        
        # Guardar modelo
        model_path = f'{output_dir}/sarima_{sku_id}.pkl'
        model.save(model_path)
        
        return model
        
    except Exception as e:
        print(f"      ❌ Error entrenando SARIMA para {sku_id}: {e}")
        return None


def evaluate_sarima_batch(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                          sample_skus=100, output_file='data/predictions_sarima.parquet'):
    """
    Evalúa modelo SARIMA en un lote de SKUs
    """
    from model_utils import temporal_train_val_test_split, calculate_metrics
    
    print(f"\n📈 Evaluando SARIMA en {sample_skus} SKUs...")
    
    # Obtener SKUs únicos
    skus = df['sku_id_comp'].unique()
    skus_sample = np.random.choice(skus, min(sample_skus, len(skus)), replace=False)
    
    results = []
    
    for i, sku in enumerate(skus_sample):
        print(f"   Procesando SKU {i+1}/{len(skus_sample)}: {sku}", end='\r')
        
        # Filtrar datos del SKU
        sku_data = df[df['sku_id_comp'] == sku].sort_values('fecha')
        
        # División temporal
        train, val, test = temporal_train_val_test_split(
            sku_data, train_ratio, val_ratio, test_ratio
        )
        
        try:
            # Entrenar con train
            model = SARIMAModel(seasonal_period=52)
            model.fit(train['demanda_real'].values, sku_id=sku)
            
            # Predecir sobre test
            test_preds = []
            for idx, row in test.iterrows():
                # Predicción one-step-ahead
                pred = model.predict(n_periods=1)[0]
                test_preds.append(max(0, int(pred)))
                
                # Actualizar modelo con nuevo dato (para siguiente predicción)
                # En un entorno real, aquí se reentrenaría periódicamente
            
            # Calcular métricas
            metrics = calculate_metrics(test['demanda_real'].values[:len(test_preds)], test_preds)
            
            results.append({
                'sku_id': sku,
                'farmacia_id': sku.split('_')[0],
                'categoria': sku_data.iloc[0]['categoria_farmaceutica'],
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'train_size': len(train),
                'test_size': len(test)
            })
            
        except Exception as e:
            print(f"\n   ⚠️ Error en SKU {sku}: {e}")
            continue
    
    # Guardar resultados
    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_file, compression='snappy')
    
    print(f"\n✅ Resultados SARIMA guardados: {output_file}")
    print(f"   MAPE promedio: {df_results['MAPE'].mean():.2f}%")
    
    return df_results