import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from models.metrics import calc_metrics
from statsmodels.tsa.seasonal import STL

def get_model(data, cross_val):
    df_prophet = data['df_serie'][['Data', 'Atendimento/Venda']].rename(columns={'Data':'ds','Atendimento/Venda':'y'})
    df_prophet_stl = df_prophet.set_index('ds')
    stl = STL(df_prophet_stl['y'], period=365) 
    res = stl.fit()
    df_prophet['seasonal'] = res.seasonal.values
    df_prophet['y_deseasonalized'] = df_prophet['y'] - df_prophet['seasonal']
    df_prophet['q4'] = df_prophet['ds'].dt.month.isin([10,11,12]).astype(int)
    n = len(df_prophet)
    train_cutoff = int(n * 0.8)  
    train_df = df_prophet.iloc[:train_cutoff]
    test_df  = df_prophet.iloc[train_cutoff:]
    df_prophet['q4'] = df_prophet['ds'].dt.month.isin([10,11,12]).astype(int)
    model = Prophet(    
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    model.add_regressor('q4')
    train_df = train_df[['ds','y_deseasonalized','q4']].rename(
        columns={'y_deseasonalized':'y'}
    )
    model.fit(train_df)
    test = test_df[['ds','q4']].copy()
    forecast = model.predict(test)
    forecast['yhat_with_seasonality'] = forecast['yhat'] + test_df['seasonal'].values
    y_pred_prophet = forecast['yhat_with_seasonality'].values
    y_test = test_df['y'].values
    metrics = calc_metrics(y_test, y_pred_prophet)
    residuos_prophet = y_test - y_pred_prophet
    erro_padrao_prophet = np.std(residuos_prophet)
    intervalo_95_prophet = 1.96 * erro_padrao_prophet
    df_resultado_prophet = pd.DataFrame({
        'data':           test_df['ds'].values, 
        'y_real':              y_test,
        'y_pred_prophet': y_pred_prophet,
        'y_lower':        y_pred_prophet - intervalo_95_prophet,
        'y_upper':        y_pred_prophet + intervalo_95_prophet
    })
    df_resultado_prophet[['y_pred_prophet','y_lower','y_upper']] = df_resultado_prophet[['y_pred_prophet','y_lower','y_upper']].fillna(0).round(0).astype(int)
    df_resultado_prophet = df_resultado_prophet.sort_index()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_resultado_prophet['data'], df_resultado_prophet["y_real"], label="Valor Real", color="blue", linewidth=2)
    ax.plot(df_resultado_prophet['data'], df_resultado_prophet["y_pred_prophet"], label="Previsão", color="green", linestyle="dashed", linewidth=2)
    ax.fill_between(df_resultado_prophet['data'], df_resultado_prophet["y_lower"], df_resultado_prophet["y_upper"], color="gray", alpha=0.3, label="Intervalo de Confiança 95%")
    ax.set_title("Previsão Prophet com Re-sazonalização e IC 95% ", fontsize=14)
    ax.set_xlabel("Tempo", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.legend()
    ax.grid(True)
    return {
        'model': model,
        'metrics': metrics,
        'plot': fig
    }