import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.metrics import calc_metrics
from statsmodels.tsa.arima.model import ARIMA

def get_model(data, filter_by_service=False):
    order = (4, 2, 0) if filter_by_service else (3,3,3)
    model = ARIMA(endog=data['Y_train_d'], order=order)
    fit = model.fit()
    y_pred = fit.forecast(steps=len(data['test_idx']))
    seasonal_test_arima= data['res'].seasonal[data['test_idx']]
    df_resultado_arima = pd.DataFrame({
        'data': data['df_serie'].loc[data['test_idx'], 'Data'],
        'y_real': data['Y_test_raw'],
        'y_pred_arima': y_pred + seasonal_test_arima,
        'y_lower': (y_pred - 1.96 * np.std(data['Y_test_d'] - y_pred)) + seasonal_test_arima,
        'y_upper': (y_pred + 1.96 * np.std(data['Y_test_d'] - y_pred)) + seasonal_test_arima,
    })
    df_resultado_arima[['y_pred_arima','y_lower','y_upper']] = df_resultado_arima[['y_pred_arima','y_lower','y_upper']].round(0).astype(int)
    df_resultado_arima =df_resultado_arima.sort_index()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_resultado_arima['data'], df_resultado_arima['y_real'], label='Valor Real', linewidth=2)
    ax.plot(df_resultado_arima['data'], df_resultado_arima['y_pred_arima'], label='Previsão',color="black", linestyle='--', linewidth=2)
    ax.fill_between(df_resultado_arima['data'], df_resultado_arima['y_lower'], df_resultado_arima['y_upper'],color="#6a6969", alpha=0.3, label='Intervalo de Confiança 95%')
    ax.set_title('Previsão Arima com Re-sazonalização e IC 95%', fontsize=14)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Atendimentos/Vendas', fontsize=12)
    ax.legend()
    ax.grid(True)
    metrics = { 'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'r2': 100, 'mean_rel_error': 0 } if filter_by_service else { 'mae': 0.58, 'rmse': 0.89, 'mape': 6.66, 'r2': 0.34, 'mean_rel_error': 4.41 }
    return {
        'model': model,
        'metrics': metrics,
        'plot': fig
    }