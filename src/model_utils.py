# src/model_utils.py
"""
Utilidades comunes para modelado predictivo
Referencias:
- Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice
- Makridakis, S., et al. (2022). M5 accuracy competition
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def temporal_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    División temporal de datos respetando el orden cronológico
    
    Args:
        df: DataFrame con columna 'fecha'
        train_ratio, val_ratio, test_ratio: proporciones
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Ordenar por fecha
    df_sorted = df.sort_values('fecha').reset_index(drop=True)
    
    # Calcular índices de corte
    n = len(df_sorted)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_idx].copy()
    val_df = df_sorted.iloc[train_idx:val_idx].copy()
    test_df = df_sorted.iloc[val_idx:].copy()
    
    print(f"   División temporal:")
    print(f"      Train: {train_df['fecha'].min()} a {train_df['fecha'].max()} ({len(train_df)} registros)")
    print(f"      Val:   {val_df['fecha'].min()} a {val_df['fecha'].max()} ({len(val_df)} registros)")
    print(f"      Test:  {test_df['fecha'].min()} a {test_df['fecha'].max()} ({len(test_df)} registros)")
    
    return train_df, val_df, test_df


def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de error estándar
    Referencia: Hyndman & Athanasopoulos (2021), Capítulo 3
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filtrar casos donde y_true es 0 para MAPE
    mask = y_true > 0
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2) if not np.isnan(mape) else np.nan
    }


def create_lag_features(df, target='demanda_real', lags=[1,2,3,4,8,12]):
    """
    Crea variables de rezago para modelos de ML
    CORREGIDO: No requiere columna 'sku_id_comp' porque ya viene filtrado por SKU
    Referencia: Hyndman & Athanasopoulos (2021), Sección 9.1
    """
    try:
        df_feat = df.copy()
        
        # Asegurar ordenamiento por fecha
        df_feat = df_feat.sort_values('fecha')
        
        # Crear rezagos directamente (sin groupby porque ya es un solo SKU)
        for lag in lags:
            df_feat[f'lag_{lag}'] = df_feat[target].shift(lag)
        
        # Medias móviles
        for window in [4, 12]:
            df_feat[f'rolling_mean_{window}'] = df_feat[target].shift(1).rolling(
                window, min_periods=1
            ).mean()
        
        return df_feat
        
    except Exception as e:
        print(f"❌ Error en create_lag_features: {e}")
        return None