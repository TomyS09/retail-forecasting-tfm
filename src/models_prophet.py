# src/models_prophet.py
"""
Modelo Prophet para predicción de demanda farmacéutica
Referencias:
- Taylor, S.J., & Letham, B. (2018). Forecasting at scale. The American Statistician
- Documentación oficial de Prophet
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pickle
import os

class ProphetModel:
    """
    Modelo Prophet con regresores adicionales
    """
    
    def __init__(self, seasonality_mode='additive', weekly_seasonality=True,
                 yearly_seasonality=True, daily_seasonality=False,
                 changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0):
        """
        Parámetros:
            seasonality_mode: 'additive' o 'multiplicative'
            changepoint_prior_scale: flexibilidad de la tendencia
            seasonality_prior_scale: fuerza de la estacionalidad
            holidays_prior_scale: fuerza de los efectos de días especiales
        """
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale
        )
        self.sku_id = None
        
    def add_holidays(self, country='EC'):
        """
        Agrega festividades del país
        Referencia: Prophet Holidays
        """
        from prophet.make_holidays import get_holiday_names, make_holidays_df
        
        # Festividades ecuatorianas
        holidays_df = make_holidays_df(
            year_list=[2023, 2024, 2025, 2026],
            country='Ecuador'
        )
        self.model.holidays = holidays_df
        
        return self
    
    def add_regressors(self, df):
        """
        Agrega regresores adicionales
        Referencia: Taylor & Letham (2018), Sección 3.2
        """
        # Regresores binarios
        for col in ['promocion', 'es_quincena', 'es_fin_semana']:
            if col in df.columns:
                self.model.add_regressor(col)
        
        return self
    
    def fit(self, df, sku_id=None):
        """
        Ajusta modelo Prophet
        """
        self.sku_id = sku_id
        
        # Preparar datos en formato Prophet
        prophet_df = df[['fecha', 'demanda_real']].rename(
            columns={'fecha': 'ds', 'demanda_real': 'y'}
        )
        
        # Agregar regresores si existen
        for col in ['promocion', 'es_quincena', 'es_fin_semana']:
            if col in df.columns:
                prophet_df[col] = df[col].values
        
        # Entrenar modelo
        self.model.fit(prophet_df)
        
        return self
    
    def predict(self, future_df):
        """
        Predice para fechas futuras
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")
        
        # Preparar dataframe futuro
        future = future_df[['fecha']].rename(columns={'fecha': 'ds'})
        
        # Agregar regresores futuros
        for col in ['promocion', 'es_quincena', 'es_fin_semana']:
            if col in future_df.columns:
                future[col] = future_df[col].values
        
        # Generar predicciones
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values
    
    def save(self, path):
        """Guarda el modelo"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """Carga un modelo guardado"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def train_prophet_for_sku(sku_data, sku_id, output_dir='models/prophet'):
    """
    Entrena modelo Prophet para un SKU específico
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ordenar por fecha
    sku_data = sku_data.sort_values('fecha')
    
    # Verificar longitud suficiente
    if len(sku_data) < 78:
        print(f"      ⚠️ SKU {sku_id}: datos insuficientes ({len(sku_data)} semanas)")
        return None
    
    try:
        # Entrenar modelo
        model = ProphetModel(
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        model.add_regressors(sku_data)
        model.fit(sku_data, sku_id=sku_id)
        
        # Guardar modelo
        model_path = f'{output_dir}/prophet_{sku_id}.pkl'
        model.save(model_path)
        
        return model
        
    except Exception as e:
        print(f"      ❌ Error entrenando Prophet para {sku_id}: {e}")
        return None


def evaluate_prophet_batch(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                           sample_skus=100, output_file='data/predictions_prophet.parquet'):
    """
    Evalúa modelo Prophet en un lote de SKUs
    """
    from model_utils import temporal_train_val_test_split, calculate_metrics
    
    print(f"\n📈 Evaluando Prophet en {sample_skus} SKUs...")
    
    skus = df['sku_id_comp'].unique()
    skus_sample = np.random.choice(skus, min(sample_skus, len(skus)), replace=False)
    
    results = []
    
    for i, sku in enumerate(skus_sample):
        print(f"   Procesando SKU {i+1}/{len(skus_sample)}: {sku}", end='\r')
        
        sku_data = df[df['sku_id_comp'] == sku].sort_values('fecha')
        
        # División temporal
        train, val, test = temporal_train_val_test_split(
            sku_data, train_ratio, val_ratio, test_ratio
        )
        
        try:
            # Entrenar con train
            model = ProphetModel()
            model.add_regressors(train)
            model.fit(train, sku_id=sku)
            
            # Predecir test (one-step-ahead rolling)
            test_preds = []
            current_train = train.copy()
            
            for idx, row in test.iterrows():
                # Crear futuro (solo la siguiente semana)
                future = pd.DataFrame([row])
                
                # Predecir
                pred = model.predict(future)[0]
                test_preds.append(max(0, int(pred)))
                
                # Actualizar entrenamiento para siguiente predicción
                # En producción, aquí se reentrenaría periódicamente
                current_train = pd.concat([current_train, pd.DataFrame([row])])
            
            # Calcular métricas
            metrics = calculate_metrics(test['demanda_real'].values, test_preds)
            
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
    
    df_results = pd.DataFrame(results)
    df_results.to_parquet(output_file, compression='snappy')
    
    print(f"\n✅ Resultados Prophet guardados: {output_file}")
    print(f"   MAPE promedio: {df_results['MAPE'].mean():.2f}%")
    
    return df_results