import streamlit as st

st.set_page_config(
    page_title="Metodologia",
    layout="wide"
)

st.sidebar.markdown("""
Desenvolvido por:
                    
ANDRESSA DOS SANTOS SILVA\n
ANDERSON PORTES DO NASCIMENTO\n
ALEXANDRE DA CUNHA FERNANDES\n
ALEXSANDRO DA SILVA BEZERRA\n
BRENO HENRIQUE REYS LORENZO\n
LEONARDO DE JESUS ANDRADE
""")

st.header("Metodologia")
st.write(
    """
Metodologia: Análise Preditiva para Gestão Estratégica\n
\n
Objetivo: Apoiar a tomada de decisões de um salão de beleza, otimizando o desempenho e maximizando lucros através de Ciência de Dados.\n
Fonte dos Dados: Entrevistas com a proprietária e base de dados históricos de vendas (em formato CSV).\n
Etapas do Projeto:\n
\n
1. Preparação e Análise Exploratória:\n
- Limpeza e tratamento da base de dados (valores faltantes, inconsistências).
- Uso de visualizações gráficas para identificar os primeiros padrões, tendências e sazonalidades nos dados históricos.
2. Modelagem e Previsão:\n
\n
Aplicação de múltiplos modelos para prever resultados futuros:
- Estatísticos: ARIMA, SARIMAX.
- Machine Learning: Prophet, XGBoost.
- Abordagens: Bayesianas.
- Otimização do modelo XGBoost através de Grid Search para máxima precisão.
3. Avaliação e Seleção:
- Comparação do desempenho dos modelos com métricas de erro (MAE, MAPE, RMSE).
- Seleção do modelo com a previsão mais acurada para ser utilizado como ferramenta estratégica.
- Tecnologia: Todo o processo foi desenvolvido em Python.
"""
)