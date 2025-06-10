import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from models import data, proph, xgb, arima, sarimax


st.set_page_config(page_title="Modelos para atendimentos totais", layout="wide")

st.sidebar.markdown("""
Desenvolvido por:
                    
ANDRESSA DOS SANTOS SILVA\n
ANDERSON PORTES DO NASCIMENTO\n
ALEXANDRE DA CUNHA FERNANDES\n
ALEXSANDRO DA SILVA BEZERRA\n
BRENO HENRIQUE REYS LORENZO\n
LEONARDO DE JESUS ANDRADE
""")

st.title("Modelos para atendimentos totais")
st.markdown("---")

selected_models = st.multiselect('Selecione os modelos: ', ['XGBOOST', 'ARIMA', 'SARIMAX', 'PROPHET'])

data = data.get_data()
if not selected_models:
    st.error('Selecione um modelo.')
    st.stop()

with st.spinner():
    models = {
        'XGBOOST': xgb.get_model(data),
        'ARIMA': arima.get_model(data),
        'SARIMAX': sarimax.get_model(data),
        'PROPHET': proph.get_model(data)
    }
    filtered_models = [{ **model, 'name': name} for [name, model] in models.items() if name in selected_models]
    df_metrics = pd.DataFrame({ 
        'Model': [m['name'] for  m in filtered_models],
        'R2 Score': [m['metrics']['r2'] for  m in filtered_models],
        'Mean Absolute Error': [m['metrics']['mae'] for  m in filtered_models],
        'Root Mean Squared Error': [m['metrics']['rmse'] for  m in filtered_models],
        'Mean Absolute Percentage Error': [m['metrics']['mape'] for  m in filtered_models],
        'Mean Relative Error': [m['metrics']['mean_rel_error'] for  m in filtered_models],
    }).set_index('Model')
    st.header('Previsões')
    for model in filtered_models:
        st.pyplot(model['plot'])
    
    st.header('Métricas')
    st.write(df_metrics)
    color_map = {
        'XGBOOST': 'orange',
        'ARIMA': 'black',
        'SARIMAX': 'red',
        'PROPHET': 'green'
    }

    for metric in df_metrics.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [color_map[model] for model in selected_models]
        ax.bar(df_metrics.index, df_metrics[metric], color=colors)
        ax.set_title(f'Métrica - {metric}')
        ax.set_xlabel('Modelo')
        ax.set_ylabel(metric)
        # ax.set_xticks(rotation=45)
        st.pyplot(fig)


