# src/models_lightgbm.py
"""
Modelo LightGBM para predicción de demanda farmacéutica
Referencias:
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- Makridakis, S., et al. (2022). M5 accuracy competition

VERSIÓN CORREGIDA SEGÚN DOCUMENTACIÓN OFICIAL:
- verbosity=-1 en parámetros para silenciar warnings
- No se necesitan loggers personalizados
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import re

from model_utils import create_lag_features, calculate_metrics

# Silenciar UserWarnings de LightGBM (opcional, pero recomendado)
warnings.filterwarnings("ignore", category=UserWarning)

def sanitize_feature_names(columns):
    """
    Sanitiza nombres de columnas para LightGBM
    Elimina/reemplaza caracteres problemáticos
    """
    sanitized = []
    for col in columns:
        # Reemplazar espacios con _
        col = re.sub(r'\s+', '_', col)
        # Eliminar caracteres especiales
        col = re.sub(r'[^\w_]', '', col)
        # Eliminar acentos (versión simple)
        col = col.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        col = col.replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
        col = col.replace('ñ', 'n').replace('Ñ', 'N')
        sanitized.append(col)
    return sanitized

class LightGBMModel:
    """
    Modelo LightGBM con validación cruzada temporal
    """
    
    def __init__(self, objective='regression', metric='mae', 
                 boosting_type='gbdt', num_leaves=31, learning_rate=0.05,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                 verbose=1, n_estimators=100, random_state=None):
        """
        Parámetros basados en:
        - Ke et al. (2017): LightGBM original
        - Makridakis et al. (2022): M5 competition settings
        
        verbose: 
            - 0 = sin salida
            - 1 = salida básica (progreso cada 100 iteraciones)
            - 2 = salida detallada
        """
        self.params = {
            'objective': objective,
            'metric': metric,
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': verbose,  # Este controla el logging de LightGBM
            'verbosity': -1,      # ← ESTO es lo que realmente silencia los warnings
            'n_estimators': n_estimators,
            'random_state': random_state if random_state is not None else 42
        }
        self.model = None
        self.feature_names = None
        self.sku_id = None
        self.verbose = verbose
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            categorical_features=None, sku_id=None):
        """
        Entrena modelo LightGBM con salida de progreso
        """
        self.sku_id = sku_id
        
        # Guardar nombres de features originales
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Asegurar que los datos son numéricos
        X_train = X_train.astype(float)
        if X_val is not None:
            X_val = X_val.astype(float)
        
        if self.verbose > 0:
            print(f"\n      🔄 LightGBM entrenando para {sku_id}...")
            print(f"         • Muestras entrenamiento: {len(X_train)}")
            if X_val is not None:
                print(f"         • Muestras validación: {len(X_val)}")
            print(f"         • Características: {len(self.feature_names) if self.feature_names else 'N/A'}")
        
        train_data = lgb.Dataset(X_train, label=y_train, 
                                 categorical_feature=categorical_features,
                                 params={"verbosity": -1})  # ← Silenciar también en Dataset
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, 
                                   categorical_feature=categorical_features,
                                   reference=train_data,
                                   params={"verbosity": -1})  # ← Silenciar también en Dataset
            
            # Callbacks personalizados
            callbacks = [lgb.early_stopping(50)]
            
            if self.verbose >= 2:
                callbacks.append(lgb.log_evaluation(10))
            else:
                callbacks.append(lgb.log_evaluation(0))
            
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[val_data],
                valid_names=['valid'],
                callbacks=callbacks
            )
            
            if self.verbose > 0:
                # LightGBM guarda la métrica como 'l1' (MAE) internamente
                best_iter = self.model.best_iteration
                best_score = self.model.best_score['valid']['l1']
                print(f"         ✓ LightGBM completado: {best_iter} iteraciones, MAE: {best_score:.4f}")
        else:
            # Sin validación
            self.model = lgb.train(
                self.params,
                train_data,
                callbacks=[lgb.log_evaluation(0)]
            )
            if self.verbose > 0:
                print(f"         ✓ LightGBM entrenado (sin validación)")
        
        return self
    
    def predict(self, X):
        """
        Predice con modelo entrenado
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")
        
        X = X.astype(float)
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def feature_importance(self, importance_type='gain'):
        """
        Retorna importancia de características
        """
        if self.model is None:
            return None
        
        importance = self.model.feature_importance(importance_type=importance_type)
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return importance
    
    def save(self, path):
        """Guarda el modelo"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """Carga un modelo guardado"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def prepare_features_for_sku(sku_data, lags=[1,2,3,4,8,12]):
    """
    Prepara características para LightGBM
    """
    try:
        # Crear variables de rezago
        df_feat = create_lag_features(sku_data, target='demanda_real', lags=lags)
        
        if df_feat is None or len(df_feat) == 0:
            print(f"   ⚠️ create_lag_features devolvió DataFrame vacío")
            return None, None
        
        # Eliminar filas con NaN
        df_feat = df_feat.dropna().copy()
        
        if len(df_feat) == 0:
            print(f"   ⚠️ Todos los registros tienen NaN después de rezagos")
            return None, None
        
        # Identificar columnas a excluir
        exclude_cols = ['fecha', 'sku_id', 'sku_id_comp', 'farmacia_id', 
                        'nombre_producto', 'presentacion', 'laboratorio',
                        'demanda_real', 'pred_arima', 'pred_prophet', 'pred_lightgbm']
        
        # One-hot encoding para variables categóricas
        categorical_cols = ['region', 'ciudad', 'tipo_farmacia', 'categoria_farmaceutica', 
                            'clasificacion_atc', 'temporada']
        
        # Guardar las columnas categóricas que existen
        existing_cat_cols = [col for col in categorical_cols if col in df_feat.columns]
        
        if existing_cat_cols:
            # Crear dummies y concatenar
            dummies = pd.get_dummies(df_feat[existing_cat_cols], prefix=existing_cat_cols)
            
            # Sanitizar nombres de las nuevas columnas
            dummies.columns = sanitize_feature_names(dummies.columns)
            
            df_feat = pd.concat([df_feat, dummies], axis=1)
            
            # Añadir las categóricas originales a exclude_cols
            exclude_cols.extend(existing_cat_cols)
        
        # Seleccionar features numéricas
        feature_cols = [col for col in df_feat.columns 
                       if col not in exclude_cols 
                       and df_feat[col].dtype in ['int64', 'float64', 'bool']]
        
        if len(feature_cols) == 0:
            print(f"   ⚠️ No se encontraron columnas de características numéricas")
            return None, None
        
        print(f"   ✓ prepare_features: {len(df_feat)} filas, {len(feature_cols)} features")
        return df_feat, feature_cols
        
    except Exception as e:
        import traceback
        print(f"   ❌ Error en prepare_features_for_sku: {str(e)}")
        print(f"      {traceback.format_exc()}")
        return None, None


def hyperparameter_tuning(X_train, y_train, X_val, y_val, n_iter=20, verbose=1):
    """
    Optimización de hiperparámetros con búsqueda aleatoria
    """
    param_dist = {
        'num_leaves': [15, 31, 50, 75, 100],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_samples': [5, 10, 20, 30, 50],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0, 0.01, 0.1, 1.0]
    }
    
    best_mae = float('inf')
    best_params = None
    
    if verbose > 0:
        print(f"      🔍 Optimizando hiperparámetros ({n_iter} iteraciones)...")
    
    for i in range(n_iter):
        params = {k: np.random.choice(v) for k, v in param_dist.items()}
        params.update({
            'objective': 'regression', 
            'metric': 'mae', 
            'verbosity': -1,  # ← Silenciar en optimización también
            'verbose': -1
        })
        
        X_train_float = X_train.astype(float)
        X_val_float = X_val.astype(float)
        
        train_data = lgb.Dataset(X_train_float, label=y_train, params={"verbosity": -1})
        val_data = lgb.Dataset(X_val_float, label=y_val, reference=train_data, params={"verbosity": -1})
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=200,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        val_pred = model.predict(X_val_float, num_iteration=model.best_iteration)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params
            
        if verbose > 0 and (i + 1) % 5 == 0:
            print(f"         • Iteración {i+1}/{n_iter} - Mejor MAE hasta ahora: {best_mae:.4f}")
    
    if verbose > 0:
        print(f"      ✓ Optimización completada - Mejor MAE: {best_mae:.4f}")
    
    return best_params