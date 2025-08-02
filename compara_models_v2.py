import pandas as pd
import numpy as np
import joblib
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor,
    HistGradientBoostingRegressor, StackingRegressor
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pickle
import warnings
import logging

warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 1. DATA INGEST, CLEANING AND IMPUTATION ==========

def load_and_merge(nrows=None):
    logging.info("Leyendo archivos de datos...")
    df_nasa = pd.read_csv('datos_centro_nasapower.csv', parse_dates=['time'], nrows=nrows)
    df_open = pd.read_csv('datos_centro_openmeteo.csv', parse_dates=['time'], nrows=nrows)
    df_ref = pd.read_excel('Data_PotenciaPEol_Historico.xlsx', nrows=nrows)
    rename_dict = {
        "Tiempo": "time",
        "WindFarm - Velocidad viento instantánea parque": "wind_speed_90m_ref",
        "Potencia activa Total": "Pot_parque",
        "Número AEGs en Marcha & Listo": "WTG_disponibles",
        "Número AEGs Limitados & en modo degradado": "WTG_invalidos"
    }
    df_ref = df_ref.rename(columns=rename_dict)
    df_ref['Pot_parque_escaled'] = 1e-3 * df_ref['Pot_parque'] / df_ref['WTG_disponibles']
    nasafeats = [c for c in df_nasa.columns if c != 'time']
    openfeats = [c for c in df_open.columns if c != 'time']
    df_nasa = df_nasa.rename(columns={c: f'{c}_nasa' for c in nasafeats})
    df_open = df_open.rename(columns={c: f'{c}_open' for c in openfeats})
    
    merged = (
        df_nasa
        .merge(df_open, on='time', how='inner')
        .merge(
            df_ref[['time', 'wind_speed_90m_ref', 'Pot_parque_escaled', 'WTG_disponibles', 'WTG_invalidos']],
            on='time', 
            how='inner'
        )
    )
   
    
    logging.info(f"Merge inicial: {len(merged)} registros.")
    return merged

def resumen_step(df, nombre):
    logging.info(f"{nombre}: {len(df)} registros, NaNs totales: {df.isnull().sum().sum()}")
    if len(df) == 0:
        logging.warning(f"¡El DataFrame está vacío después de {nombre}! Revisa este paso.")
    elif len(df) < 50:
        logging.warning(f"Quedan pocos registros ({len(df)}) después de {nombre}. ¡Posible sobre-filtrado!")
    return df

def imputar_iterativo_condicional(df, cols_a_imputar, col_flag_zero='WTG_invalidos', random_state=42):
    df = df.copy()
    mask_nan = df[cols_a_imputar].isnull().any(axis=1)
    mask_zero = df[col_flag_zero] == 0 if col_flag_zero in df.columns else np.array([False] * len(df))
    mask_impute = mask_nan | mask_zero
    if not mask_impute.any():
        logging.info("No se requiere imputación iterativa (no hay NaNs ni WTG_invalidos==0).")
        return df
    df_to_impute = df.loc[mask_impute, cols_a_imputar].replace([np.inf, -np.inf], np.nan)
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=random_state), max_iter=10, random_state=random_state)
    imputed_array = imputer.fit_transform(df_to_impute)
    df.loc[mask_impute, cols_a_imputar] = imputed_array
    logging.info(f"Imputación iterativa completada para {mask_impute.sum()} filas.")
    return df

def filtra_por_iqr(df, cols, k=1.5):
    def max_diff(row):
        vals = [row[c] for c in cols if not pd.isnull(row[c])]
        if len(vals) < 2:
            return 0
        diffs = [abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i+1, len(vals))]
        return max(diffs)
    dif_max = df.apply(max_diff, axis=1)
    Q1 = np.percentile(dif_max, 25)
    Q3 = np.percentile(dif_max, 75)
    IQR = Q3 - Q1
    lim_inf = Q1 - k * IQR
    lim_sup = Q3 + k * IQR
    mask = (dif_max >= lim_inf) & (dif_max <= lim_sup)
    logging.info(f"Filtrado por IQR: quedan {mask.sum()} registros de {len(df)}.")
    return df[mask]

def limpieza_extra(df, cols, sigma=3.0):
    # Primero aplicar el filtro z-score
    z_scores = np.abs(zscore(df[cols], nan_policy='omit'))
    mask_z = (z_scores < sigma).all(axis=1)
    df = df[mask_z]
    logging.info(f"Limpieza extra (z-score): quedan {len(df)} registros de {len(df) + (~mask_z).sum()}.")
    
    # Filtrado adicional de valores fuera de tendencia
    # Filtro 1: Si wind_speed_90m_ref < 4, eliminar filas con Pot_parque > 1
    df = df[~((df['wind_speed_90m_ref'] < 4) & (df['Pot_parque_escaled'] > 1))]

    # Filtro 2: Si wind_speed_90m_ref < 6, eliminar filas con Pot_parque > 2
    df = df[~((df['wind_speed_90m_ref'] < 6) & (df['Pot_parque_escaled'] > 2))]

    # Filtro 3: Si wind_speed_90m_ref < 8, eliminar filas con Pot_parque > 4
    df = df[~((df['wind_speed_90m_ref'] < 8) & (df['Pot_parque_escaled'] > 4))]

    # Filtro 4: Si wind_speed_90m_ref > 10.5, eliminar filas con Pot_parque < 1
    df = df[~((df['wind_speed_90m_ref'] > 12.5) & (df['Pot_parque_escaled'] < 3))]

    # Filtro 5: Si wind_speed_90m_ref > 12.5, eliminar filas con Pot_parque < 2
    df = df[~((df['wind_speed_90m_ref'] > 12.5) & (df['Pot_parque_escaled'] < 3))]

    # Filtro 6: Si wind_speed_90m_ref > 14, eliminar filas con Pot_parque < 3
    df = df[~((df['wind_speed_90m_ref'] > 14) & (df['Pot_parque_escaled'] < 4))]

    # Filtro 7: Valores negativos de Pot_parque_escaled y valores mayores de 40
    df = df[df['Pot_parque_escaled'] >= 0]
    df = df[df['Pot_parque_escaled'] <= 6.0]
       
    return df.dropna(subset=cols)

def add_features(df):
    df = df.copy()
    for feat in ['wind_speed_90m_nasa', 'wind_speed_90m_open']:
        df[f'{feat}_sq'] = df[feat]**2
        df[f'{feat}_cub'] = df[feat]**3
        df[f'{feat}_lag1'] = df[feat].shift(1)
        df[f'{feat}_lag24'] = df[feat].shift(24)
        df[f'{feat}_rollmean3'] = df[feat].rolling(3, min_periods=1).mean()
        df[f'{feat}_rollstd3'] = df[feat].rolling(3, min_periods=1).std()
    for pre in ['temperature_2m', 'surface_pressure', 'pressure_90m']:
        for suf in ['nasa', 'open']:
            c = f"{pre}_{suf}"
            if c in df.columns:
                df[c] = df[c].interpolate().fillna(method='bfill').fillna(method='ffill')
    df['delta_wind90'] = df['wind_speed_90m_nasa'] - df['wind_speed_90m_open']
    df['product_wind90'] = df['wind_speed_90m_nasa'] * df['wind_speed_90m_open']
    return df


def analizar_importancia_features(best_model_name, features_sel):
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt

    # Si es ensemble, combina importancia de cada modelo base según su peso
    if best_model_name.startswith('Ensemble'):
        info = joblib.load('ensemble_model_info.pkl')
        names, weights = info['names'], np.array(info['weights'])
        feature_importances = np.zeros(len(features_sel))
        # Importancia por cada modelo base, promediando
        for n, w in zip(names, weights):
            model = joblib.load(f"model_{n}.pkl")
            # Si el modelo tiene feature_importances_
            if hasattr(model, "feature_importances_"):
                fi = np.array(model.feature_importances_)
            # Si es un MLP o modelo sin importance, usa Permutation Importance
            else:
                try:
                    from sklearn.inspection import permutation_importance
                    X_sample = np.random.rand(50, len(features_sel))
                    y_sample = np.random.rand(50)
                    result = permutation_importance(model, X_sample, y_sample, n_repeats=2, random_state=42)
                    fi = result.importances_mean
                except Exception as e:
                    print(f"No se pudo calcular importance para {n}: {e}")
                    fi = np.zeros(len(features_sel))
            feature_importances += w * fi
        feature_importances /= weights.sum()
    else:
        # Carga modelo individual
        model = joblib.load(f"best_model_{best_model_name}.pkl")
        if hasattr(model, "feature_importances_"):
            feature_importances = np.array(model.feature_importances_)
        else:
            # Para MLP u otros, usa permutation importance sobre datos sintéticos
            from sklearn.inspection import permutation_importance
            X_sample = np.random.rand(50, len(features_sel))
            y_sample = np.random.rand(50)
            result = permutation_importance(model, X_sample, y_sample, n_repeats=2, random_state=42)
            feature_importances = result.importances_mean

    # Muestra la importancia ordenada
    order = np.argsort(-feature_importances)
    plt.figure(figsize=(10,4))
    plt.bar(np.array(features_sel)[order], feature_importances[order])
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Importancia de Features en el Modelo: {best_model_name}")
    plt.ylabel("Importancia relativa")
    plt.tight_layout()
    plt.show()
    # Imprime tabla ordenada
    print("\nRanking de features por importancia (de mayor a menor):")
    for feat, imp in zip(np.array(features_sel)[order], feature_importances[order]):
        print(f"  {feat:30s} {imp:.4f}")



# ========== 2. GRÁFICO CURVA MEDIA/DISPERSIÓN ANTES DE LIMPIAR ==========

def grafico_curva_media(df, x_col='wind_speed_90m_ref', 
                        y_col='Pot_parque_escaled', 
                        bin_width=0.5,
                        titulo=''):
    bins = np.arange(df[x_col].min(), df[x_col].max() + bin_width, bin_width)
    df['bin_x'] = pd.cut(df[x_col], bins)
    media_y = df.groupby('bin_x')[y_col].mean()
    centros_bin = [interval.mid for interval in media_y.index]

    plt.figure(figsize=(8,5))
    plt.scatter(df[x_col], df[y_col], alpha=0.3, label='Datos reales')
    plt.plot(centros_bin, media_y.values, color='red', linewidth=2, linestyle='--', label='Curva media')
    plt.xlabel(f'{x_col} [unidades]')
    plt.ylabel(f'{y_col}')
    plt.title(f'Curva media: {y_col} vs {x_col}'+titulo)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Limpia la columna temporal para que no se propague
    df.drop(columns=['bin_x'], inplace=True)

# ========== 3. NORMALIZACIÓN MIN-MAX DE FEATURES ==========

def normalizar_features(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler

def revertir_normalizacion(X_norm, scaler):
    return scaler.inverse_transform(X_norm)

# ========== 4. SELECCIÓN AUTOMÁTICA DE FEATURES ==========

def feature_selection(X, y, features, top_k=20, corr_thres=0.95, do_pca=False, verbose=True):
    logging.info(f"Entrando a selección automática de features con {len(features)} columnas originales.")
    # 1. Elimina columnas con correlación alta
    df_feats = pd.DataFrame(X, columns=features)
    corr_matrix = df_feats.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > corr_thres)]
    if drop_cols:
        logging.info(f"Dropping {len(drop_cols)} columns with correlation > {corr_thres}: {drop_cols}")
    X = np.delete(X, [features.index(c) for c in drop_cols], axis=1)
    features = [f for f in features if f not in drop_cols]
    # 2. SelectFromModel con RandomForest (importancia > mediana)
    sfm = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")
    sfm.fit(X, y)
    X_new = sfm.transform(X)
    feats_selected = [f for f, keep in zip(features, sfm.get_support()) if keep]
    if verbose:
        logging.info(f"SelectFromModel mantiene {len(feats_selected)}/{len(features)} features")
    # 3. Permutation importance para top_k
    if len(feats_selected) > top_k:
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_new, y)
        result = permutation_importance(rf, X_new, y, n_repeats=5, random_state=42)
        top_idx = np.argsort(result.importances_mean)[-top_k:]
        X_new = X_new[:, top_idx]
        feats_selected = [feats_selected[i] for i in top_idx]
        logging.info(f"Permutation Importance: se seleccionan top {len(feats_selected)} features por importancia.")
    # (PCA opcional, aquí omitido para claridad)
    return X_new, feats_selected

# ========== 5. FUNCIONES DE MÉTRICAS, PIPELINE Y PRINCIPAL ==========

def calcular_metricas(y_true, y_pred, baseline=None):
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return {m: np.nan for m in ['MAE', 'RMSE', 'R2', 'MAPE', 'MedAE', 'SkillScore']}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    medae = median_absolute_error(y_true, y_pred)
    if baseline is None:
        baseline = np.roll(y_true, 1)
        baseline[0] = y_true[0]
    mae_baseline = mean_absolute_error(y_true, baseline)
    skill = 1 - (mae / mae_baseline)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "MedAE": medae, "SkillScore": skill}

def safe_metricas(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask_valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask_valid) == 0:
        logging.warning("⚠️  No hay datos válidos para métricas (solo NaN).")
        return {m: np.nan for m in ['MAE', 'RMSE', 'R2', 'MAPE', 'MedAE', 'SkillScore']}
    y_true_valid = y_true[mask_valid]
    y_pred_valid = y_pred[mask_valid]
    return calcular_metricas(y_true_valid, y_pred_valid)

def graficar_residuos(y_true, y_pred, nombre_modelo=''):
    resid = y_true - y_pred
    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    plt.scatter(y_pred, resid, alpha=0.3, color='royalblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicción')
    plt.ylabel('Residuo')
    plt.title(f'Residuos vs. Predicción\n({nombre_modelo})')
    plt.grid(ls=':')
    plt.subplot(1,2,2)
    plt.hist(resid, bins=30, color='deepskyblue', alpha=0.7, edgecolor='k')
    plt.title(f'Histograma de residuos\n({nombre_modelo})')
    plt.xlabel('Residuo')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()



# ========== 3. NUEVAS FUNCIONES PARA GUARDAR Y VALIDAR ==========

def guardar_mejor_modelo(best_model_name, models_dict, scaler_X, features_sel, resumen_path='resumen_modelos_ensemble.csv'):
    """Guarda el mejor modelo y scaler para producción"""
    resumen = pd.read_csv(resumen_path)
    best_row = resumen[resumen['Modelo'] == best_model_name].iloc[0]
    if best_model_name.startswith('Ensemble'):
        names = eval(best_row['Combinación'])
        weights = eval(best_row['Pesos'])
        joblib.dump({'names': names, 'weights': weights, 'features': features_sel, 'scaler': scaler_X}, 'ensemble_model_info.pkl')
        print(f"\nComponentes y receta del ensemble guardados en ensemble_model_info.pkl")
        # Guardar modelos base por si se requieren para despliegue
        for n in names:
            joblib.dump(models_dict[n], f"model_{n}.pkl")
        print(f"Modelos individuales del ensemble guardados.")
    else:
        joblib.dump(models_dict[best_model_name], f"best_model_{best_model_name}.pkl")
        joblib.dump({'scaler': scaler_X, 'features': features_sel}, 'scaler_X.pkl')
        print(f"Modelo {best_model_name} y scaler guardados para producción.")

def validar_modelo_produccion(X_test, y_test, best_model_name):
    """Valida el mejor modelo guardado sobre el set de test"""
    print(f"\n=== VALIDACIÓN FINAL DEL MODELO EN TEST ===")
    if best_model_name.startswith('Ensemble'):
        info = joblib.load('ensemble_model_info.pkl')
        names, weights = info['names'], info['weights']
        modelos = [joblib.load(f"model_{n}.pkl") for n in names]
        preds = np.zeros_like(y_test)
        for model, w in zip(modelos, weights):
            preds += w * model.predict(X_test)
    else:
        model = joblib.load(f"best_model_{best_model_name}.pkl")
        preds = model.predict(X_test)
    # Usa tu función de métricas y gráfico
    met = safe_metricas(y_test, preds)
    print(f"Métricas de validación final:")
    for k, v in met.items():
        print(f"  {k}: {v:.4f}")
    graficar_residuos(y_test, preds, best_model_name)


# ========== 4. PREDICCIÓN HORARIA DEL DÍA SIGUIENTE ==========
def prediccion_horaria_dia_siguiente(merged, features_sel, best_model_name):
    """
    Toma 48h del set (24h + 24h siguiente), produce features y predice potencia del día siguiente
    usando el modelo ganador. Mide error real vs target.
    Supone que merged[features_sel] YA está normalizado.
    """
    merged = merged.reset_index(drop=True)
    n_total = len(merged)
    idx_ini = n_total - 48  # últimas 48h del set limpio
    df_48h = merged.iloc[idx_ini:idx_ini+48].copy()

    # No recalcules features si ya los tienes, solo usa features_sel
    X_pred = df_48h[features_sel].copy()    # <-- usa datos YA normalizados

    # --- 4. Selecciona solo las filas del día siguiente (horas 24 a 47) ---
    X_pred_nextday = X_pred.iloc[24:48].values  # si X_pred es DataFrame
    times_nextday = df_48h['time'].values[24:48]
    y_true_nextday = df_48h['Pot_parque_escaled'].values[24:48]

    # --- 5. Cargar modelo ganador y predecir ---
    if best_model_name.startswith('Ensemble'):
        import joblib
        info = joblib.load('ensemble_model_info.pkl')
        names, weights = info['names'], info['weights']
        modelos = [joblib.load(f"model_{n}.pkl") for n in names]
        y_pred = np.zeros(len(X_pred_nextday))
        for m, w in zip(modelos, weights):
            y_pred += w * m.predict(X_pred_nextday)
    else:
        import joblib
        model = joblib.load(f"best_model_{best_model_name}.pkl")
        y_pred = model.predict(X_pred_nextday)

    # --- 6. Métricas y gráfico ---
    met = safe_metricas(y_true_nextday, y_pred)
    print(f"\n=== PREDICCIÓN DÍA SIGUIENTE SOBRE SET DE TEST ===")
    print("Fechas predichas:", pd.to_datetime(times_nextday))
    print("Métricas reales en el día siguiente:")
    for k, v in met.items():
        print(f"  {k}: {v:.4f}")
    graficar_residuos(y_true_nextday, y_pred, f"Predicción Día Siguiente: {best_model_name}")

    # --- 7. Tabla de comparación ---
    df_result = pd.DataFrame({
        "time": pd.to_datetime(times_nextday),
        "pot_real": y_true_nextday,
        "pot_pred": y_pred
    })
    print(df_result)
    df_result.to_csv('prediccion_horaria_dia_siguiente.csv', index=False)
    print("Resultados guardados como prediccion_horaria_dia_siguiente.csv")




# ========== 4. PIPELINE COMPLETO ACTUALIZADO ==========

def pipeline_completo(
    nombre, X_norm, y, features_sel, print_graphics=True
):
    logging.info(f"Pipeline: {nombre}, registros: {len(X_norm)}")
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, shuffle=False)
    results, preds_dict, models_dict, metrics_dict = [], {}, {}, {}

    # MODELOS
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    search_spaces = {
        'RandomForest': {'n_estimators': [80,120,180,250], 'max_depth': [3,5,8,None]},
        'GradientBoosting': {'n_estimators': [80,120,180], 'learning_rate': [0.03,0.07,0.1], 'max_depth': [3,5,8]},
        'MLP': {'hidden_layer_sizes': [(60,20),(80,40),(100,50)], 'alpha':[0.001,0.01,0.1],'max_iter': [500, 1000]},
        'XGBRegressor': {'n_estimators': [60,100,180], 'learning_rate': [0.03,0.07,0.1], 'max_depth':[3,5,8]},
        'LGBMRegressor': {'n_estimators': [60,100,180], 'learning_rate': [0.03,0.07,0.1], 'max_depth':[3,5,8]},
        'CatBoostRegressor': {'iterations': [60,100,180], 'learning_rate': [0.03,0.07,0.1], 'depth':[3,5,8]}
    }
    modelos = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Bagging": BaggingRegressor(n_estimators=100, random_state=42),
        "HistGB": HistGradientBoostingRegressor(random_state=42),
        "MLP": MLPRegressor(max_iter=300, random_state=42),
        "XGB": XGBRegressor(verbosity=0, random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbosity=-1, log_level='fatal'),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
    }

    # LINEAR REGRESSION BASE
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    preds_dict['LinearRegression'] = y_pred
    models_dict['LinearRegression'] = model
    metrics_dict['LinearRegression'] = safe_metricas(y_test, y_pred)
    results.append(['LinearRegression', metrics_dict['LinearRegression']['R2'], metrics_dict['LinearRegression']['MAE']])

    # MODELOS AVANZADOS
    for name, model in modelos.items():
        params = search_spaces.get(name, None) or search_spaces.get(model.__class__.__name__, None)
        X_train_used = X_train
        y_train_used = y_train
        fit_params = {}
        if name in ["XGB", "LightGBM"] and params is None:
            n_val = int(0.2 * len(X_train))
            X_train_used = X_train[:-n_val]
            y_train_used = y_train[:-n_val]
            X_val = X_train[-n_val:]
            y_val = y_train[-n_val:]
            fit_params = {'early_stopping_rounds': 20, 'eval_set': [(X_val, y_val)], 'verbose': False}
        if params is not None:
            search = RandomizedSearchCV(
                model, params, n_iter=10, cv=3, 
                n_jobs=-1, scoring='r2', random_state=42
            )
            search.fit(X_train_used, y_train_used)
            model = search.best_estimator_
        else:
            model.fit(X_train_used, y_train_used, **fit_params)
        y_pred = model.predict(X_test)
        preds_dict[name] = y_pred
        models_dict[name] = model
        metrics_dict[name] = safe_metricas(y_test, y_pred)
        results.append([name, metrics_dict[name]['R2'], metrics_dict[name]['MAE']])

    # Stacking ensemble
    stack = StackingRegressor(
        estimators=[(k, modelos[k]) for k in ['RandomForest', 'GradientBoosting', 'MLP', 'XGB']],
        final_estimator=LinearRegression(), n_jobs=-1
    )
    stack.fit(X_train, y_train)
    y_pred_stack = stack.predict(X_test)
    preds_dict['Stacking'] = y_pred_stack
    models_dict['Stacking'] = stack
    metrics_dict['Stacking'] = safe_metricas(y_test, y_pred_stack)
    results.append(['Stacking', metrics_dict['Stacking']['R2'], metrics_dict['Stacking']['MAE']])

    # Weighted average ensemble (top 5)
    from itertools import combinations, product
    def best_weighted_ensemble(preds_dict, y_true, N=2):
        best_score, best_weights, best_names, best_pred = -np.inf, None, None, None
        keys = list(preds_dict.keys())
        for subset in combinations(keys, N):
            preds = [preds_dict[name] for name in subset]
            steps = np.arange(0,1.05,0.1)
            for weights in product(steps, repeat=N):
                if abs(sum(weights)-1)>1e-4: continue
                ens = sum(w*p for w,p in zip(weights, preds))
                min_len = min(len(y_true), len(ens))
                y_true_valid = y_true[:min_len]
                ens_valid = ens[:min_len]
                mask_valid = ~np.isnan(y_true_valid) & ~np.isnan(ens_valid)
                if not np.any(mask_valid):
                    continue
                y_true_valid = y_true_valid[mask_valid]
                ens_valid = ens_valid[mask_valid]
                if len(y_true_valid) == 0 or np.all(ens_valid == ens_valid[0]):
                    continue
                r2 = r2_score(y_true_valid, ens_valid)
                if r2 > best_score:
                    best_score, best_weights, best_names, best_pred = r2, weights, subset, ens
        return best_names, best_weights, best_score, best_pred

    top_N = 6
    top_keys = [x[0] for x in sorted(results, key=lambda x: -x[1])[:top_N] if x[0] in preds_dict]
    results_ens = []
    for N in [2,3,4]:
        keysN = {k:preds_dict[k] for k in top_keys}
        names, weights, r2, pred = best_weighted_ensemble(keysN, y_test, N)
        if pred is None:
            continue
        mae = mean_absolute_error(y_test[~np.isnan(pred)], pred[~np.isnan(pred)])
        metrics_dict[f"Ensemble_{N}"] = safe_metricas(y_test, pred)
        results_ens.append([f"Ensemble_{N}", r2, mae, names, weights])
        preds_dict[f"Ensemble_{N}"] = pred

    # Ranking final
    results_extended = [r + ['', ''] for r in results]
    results_extended += [[ens[0], ens[1], ens[2], str(ens[3]), str(ens[4])] for ens in results_ens]
    summary_df = pd.DataFrame(results_extended, columns=['Modelo','R2','MAE','Combinación','Pesos'])
    summary_df = summary_df.sort_values('R2', ascending=False).reset_index(drop=True)
    idx_best = summary_df['R2'].astype(float).idxmax()
    best_model_name = summary_df.loc[idx_best, 'Modelo']
    print("\n--- Modelos y Ensembles ---")
    print(summary_df.to_string(index=False, float_format="%.4f"))
    summary_df.to_csv('resumen_modelos_ensemble.csv', index=False)
    print('Resumen de modelos guardado como resumen_modelos_ensemble.csv')

    # Métricas detalladas y residuos
    for model in summary_df['Modelo']:
        if model in preds_dict:
            print(f"\nMétricas para modelo {model}:")
            met = metrics_dict[model]
            for k, v in met.items():
                print(f"  {k}: {v:.4f}")
            if print_graphics:
                graficar_residuos(y_test, preds_dict[model], model)

    print(f"\nMejor modelo: {best_model_name}, R²={summary_df.loc[idx_best,'R2']:.4f}, MAE={summary_df.loc[idx_best,'MAE']:.3f}")

    # --- NUEVO: Devolver todo lo necesario ---
    return {
        "Pipeline": nombre,
        "Num_muestras": len(X_norm),
        "Mejor_2inputs": best_model_name,
        "R2_2inputs": summary_df.loc[idx_best, 'R2'],
        "MAE_2inputs": summary_df.loc[idx_best, 'MAE'],
        "models_dict": models_dict,
        "preds_dict": preds_dict,
        "scaler_X": None,  # se rellena después
        "features_sel": features_sel,
        "X_test": X_test,
        "y_test": y_test
    }


# ========== 5. MAIN PRINCIPAL ACTUALIZADO ==========

def main(
    nrows=1000,
    top_k_features=20,
    correlation_threshold=0.96,
    print_graphics=True,
    imputar = False,
    target_variable= 'Pot_parque_escaled'   
):
    merged = load_and_merge(nrows=nrows)
    resumen_step(merged, "Datos originales (merge inicial)")
    merged = add_features(merged)

    # --- Gráfico solicitado: antes de limpiar ni normalizar ---
    grafico_curva_media(
        merged, 
        x_col='wind_speed_90m_ref', 
        y_col='Pot_parque_escaled',
        bin_width=1.5,
        titulo=' \n Antes de Limpiar ni normalizar'
    )

    features_finales = [c for c in merged.columns if c not in ['time', target_variable]]
    cols = [c for c in features_finales if c.startswith('wind_speed_90m_') or c.startswith('Pot_parque') or c.startswith('WTG_')]

    if imputar: 
        merged = imputar_iterativo_condicional(merged, [col for col in cols if col in merged.columns], col_flag_zero='WTG_invalidos')
        resumen_step(merged, "Después de imputación iterativa")
  
    merged = merged.dropna(subset=cols)
    
    resumen_step(merged, "Después de dropna en cols clave")
    merged = filtra_por_iqr(merged, cols, k=1.5)
    resumen_step(merged, "Después de filtra_por_iqr")
 
    merged = limpieza_extra(merged, cols, sigma=3.0)
    resumen_step(merged, "Después de limpieza_extra (z-score)")
 
    is_finite = np.isfinite(merged[features_finales + [target_variable]]).all(axis=1)
    merged = merged.loc[is_finite].reset_index(drop=True)
    resumen_step(merged, "Después de eliminar inf y NaN en features_finales+target")
 
    resumen_step(merged, "Después de dropna en features_finales+target")
    print(f"Registros finales tras limpieza: {len(merged)}")
    if len(merged) == 0:
        raise ValueError("¡DataFrame merged quedó vacío tras el preprocesamiento! Revisa tus filtros o archivos.")

    # Normalización
    X = merged[features_finales].values
    X_norm, scaler_X = normalizar_features(X)
    y = merged[target_variable].values  # Solo target nominal

    # Selección automática de features (colinealidad, importancia)
    X_sel, features_sel = feature_selection(
        X_norm, y, features_finales, top_k=top_k_features, corr_thres=correlation_threshold
    )
    logging.info(f"Features finales tras selección: ({len(features_sel)}) {features_sel}")
    
    
    # Construir el DataFrame normalizado y reducido con los features seleccionados
    merged_sel = merged[['time', 'Pot_parque_escaled']].copy()  # time y target
    # Agrega las columnas normalizadas y seleccionadas
    for i, f in enumerate(features_sel):
        merged_sel[f] = X_sel[:, i]
    
    
    # Guarda el scaler para revertir en producción
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler_X, 'features': features_sel}, f)
    print("Scaler de features guardado como scaler_X.pkl")

    # --- Gráfico solicitado: despúes de limpiar ni normalizar ---
    grafico_curva_media(
        merged, 
        x_col='wind_speed_90m_ref', 
        y_col='Pot_parque_escaled',
        bin_width=1.5,
        titulo='\n Despúes de Limpiar'
    )

    # Ejecuta pipeline, ahora devuelve todos los objetos
    resultado = pipeline_completo(
        nombre="Resultados",
        X_norm=X_sel,
        y=y,
        features_sel=features_sel,
        print_graphics=print_graphics
    )
    resultado["scaler_X"] = scaler_X  # Guarda el scaler para guardar

    print("\n==== COMPARATIVA FINAL ====")
    df_final = pd.DataFrame([resultado])
    print(df_final[['Pipeline','Num_muestras','Mejor_2inputs','R2_2inputs','MAE_2inputs']].to_string(index=False))
    best_model_name = resultado['Mejor_2inputs']

    # Guardar mejor modelo y scaler
    guardar_mejor_modelo(
        best_model_name=best_model_name,
        models_dict=resultado['models_dict'],
        scaler_X=resultado['scaler_X'],
        features_sel=resultado['features_sel']
    )

    # Validar desempeño en test (usando modelo guardado)
    validar_modelo_produccion(
        X_test=resultado['X_test'],
        y_test=resultado['y_test'],
        best_model_name=best_model_name
    )

    logging.info("Pipeline execution completed. Ready for daily prediction script.")
    
   
    # ========== 6. DETECCIÓN DE ANOMALÍAS CON DBSCAN EN RESIDUOS + FEATURES ==========
    # 1. Recupera X_test y residuos del modelo ganador
    X_test = resultado['X_test']
    y_test = resultado['y_test']
    best_model_name = resultado['Mejor_2inputs']
    y_pred = resultado['preds_dict'][best_model_name]
    residuos = y_test - y_pred

    # 2. Opcional: Combina features y residuos para el clustering
    # Puedes probar solo features, solo residuos, o ambos juntos como aquí:
    X_clust = np.hstack([X_test, residuos.reshape(-1,1)])

    # 3. Estandariza antes de DBSCAN (recomendado)
    scaler = StandardScaler()
    X_clust_scaled = scaler.fit_transform(X_clust)

    # 4. Aplica PCA a 2D para visualizar
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_clust_scaled)

    # 5. Ajusta DBSCAN (puedes ajustar eps y min_samples)
    dbscan = DBSCAN(eps=2.0, min_samples=20)  # Ajusta eps según tus datos
    labels = dbscan.fit_predict(X_clust_scaled)

    # 6. Visualización de clusters (y anomalías)
    plt.figure(figsize=(7,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.title('DBSCAN en residuos + features (espacio PCA)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster DBSCAN (-1 = anomalía)')
    plt.show()

    # 7. Analiza las anomalías (label = -1)
    anomalies = (labels == -1)
    print(f"Se detectaron {anomalies.sum()} anomalías con DBSCAN sobre {len(labels)} muestras de test ({100*anomalies.mean():.2f}%)")

    # 8. Puedes ver cómo se distribuyen los residuos para anomalías
    plt.figure()
    plt.hist(residuos[anomalies], bins=30, color='red', alpha=0.6, label='Anomalías')
    plt.hist(residuos[~anomalies], bins=30, color='blue', alpha=0.4, label='No anomalías')
    plt.xlabel('Residuo')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.title('Distribución de residuos (Anomalías vs No Anomalías, DBSCAN)')
    plt.show()
    
    # Análisis de importancia del modelo ganador:
    analizar_importancia_features(
        resultado['Mejor_2inputs'],
        resultado['features_sel']
    )
    
    # Predicción horaria del día siguiente
    prediccion_horaria_dia_siguiente(
        merged=merged_sel,  # O el nombre de tu DataFrame normalizado con features listos
        features_sel=features_sel,
        best_model_name=best_model_name
    )    
    # ========== 6. ENTRYPOINT ==========

if __name__ == "__main__":
    main(
        nrows=None,  # None para cargar todo el dataset
        top_k_features=20, # Número de features a seleccionar
        correlation_threshold=0.96,  # Umbral de correlación para eliminar features
        print_graphics=False,
        imputar = False,
        target_variable= 'Pot_parque_escaled'
    )
    
    
