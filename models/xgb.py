import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from models.metrics import calc_metrics

def get_model(data, cross_val):
    best_params = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.01
    }
    model = XGBRegressor(**best_params, random_state=42, use_label_encoder=False)
    model.fit(data['X'], data['Y_deseason'])
    seasonal_test_xgboost= data['res'].seasonal[data['test_idx']]
    y_pred_xgboost = model.predict(data['X_test'])
    metrics_xgb = calc_metrics(data['Y_test_d'], y_pred_xgboost)
    df_resultado_xgboost = pd.DataFrame({
        'data': data['df_serie'].loc[data['test_idx'], 'Data'],
        'y_real': data['Y_test_raw'],
        'y_pred_xgboost': y_pred_xgboost + seasonal_test_xgboost,
        'y_lower': (y_pred_xgboost - 1.96 * np.std(data['Y_test_d'] - y_pred_xgboost)) + seasonal_test_xgboost,
        'y_upper': (y_pred_xgboost + 1.96 * np.std(data['Y_test_d'] - y_pred_xgboost)) + seasonal_test_xgboost,
    })
    df_resultado_xgboost[['y_pred_xgboost','y_lower','y_upper']] = df_resultado_xgboost[['y_pred_xgboost','y_lower','y_upper']].round(0).astype(int)
    df_resultado_xgboost = df_resultado_xgboost.sort_index()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_resultado_xgboost['data'], df_resultado_xgboost['y_real'], label='Valor Real', linewidth=2)
    ax.plot(df_resultado_xgboost['data'], df_resultado_xgboost['y_pred_xgboost'], label='Previsão',color = "orange",linestyle='--', linewidth=2)
    ax.fill_between(df_resultado_xgboost['data'], df_resultado_xgboost['y_lower'], df_resultado_xgboost['y_upper'],color="#6a6969", alpha=0.3, label='Intervalo de Confiança 95%')
    ax.set_title('Previsão XGBoost com Re-sazonalização e IC 95%', fontsize=14)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Atendimentos/Vendas', fontsize=12)
    ax.legend()
    ax.grid(True)
    return {
        'model': model,
        'metrics': metrics_xgb,
        'plot': fig
    }
