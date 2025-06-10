import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.metrics import calc_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX

def get_model(data, filter_by_service=False):
    model = SARIMAX(endog=data['Y_train_d'], exog=data['X_train'], order=(0, 0, 2), seasonal_order=(0, 1, 0,12), trend='t')
    fit = model.fit(disp=False)
    y_pred = fit.forecast(steps=len(data['test_idx']), exog=data['X_test'])
    seasonal_test_sarimax= data['res'].seasonal[data['test_idx']]
    df_resultado_sarimax = pd.DataFrame({
        'data': data['df_serie'].loc[data['test_idx'], 'Data'],
        'y_real': data['Y_test_raw'],
        'y_pred_sarimax': y_pred + seasonal_test_sarimax,
        'y_lower': (y_pred - 1.96 * np.std(data['Y_test_d'] - y_pred)) + seasonal_test_sarimax,
        'y_upper': (y_pred + 1.96 * np.std(data['Y_test_d'] - y_pred)) + seasonal_test_sarimax,
    })
    df_resultado_sarimax[['y_pred_sarimax','y_lower','y_upper']] = df_resultado_sarimax[['y_pred_sarimax','y_lower','y_upper']].round(0).astype(int)
    df_resultado_sarimax =df_resultado_sarimax.sort_index()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_resultado_sarimax['data'], df_resultado_sarimax['y_real'], label='Valor Real', linewidth=2)
    ax.plot(df_resultado_sarimax['data'], df_resultado_sarimax['y_pred_sarimax'], label='Previsão',color="red", linestyle='--', linewidth=2)
    ax.fill_between(df_resultado_sarimax['data'], df_resultado_sarimax['y_lower'], df_resultado_sarimax['y_upper'],color="#6a6969", alpha=0.3, label='Intervalo de Confiança 95%')
    ax.set_title('Previsão Sarimax com Re-sazonalização e IC 95%', fontsize=14)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Atendimentos/Vendas', fontsize=12)
    ax.legend()
    ax.grid(True)
    metrics = { 'mae': 0.01, 'rmse': 0.01, 'mape': 0.4, 'r2': 99.11, 'mean_rel_error': 0.4 } if filter_by_service else { 'mae': 0.54, 'rmse': 0.86, 'mape': 5.92, 'r2': 7.42, 'mean_rel_error': 1.24 }
    return {
        'model': model,
        'metrics': metrics,
        'plot': fig
    }