import streamlit as st

# --- Configuração da Página (deve ser o primeiro comando) ---
st.set_page_config(
    page_title="Aprimoramento e Validação de Modelos para Séries Temporais",
    layout="centered"
)

# --- Conteúdo da Página Principal ---
st.title("Aprimoramento e Validação de Modelos para Séries Temporais")
st.markdown("---")

st.header("Introdução")
st.write(
    """
Este projeto tem como objetivo principal avaliar e aprimorar os modelos
preditivos aplicados anteriormente para estimar o número de atendimentos e a demanda
por serviço específico em um salão de beleza. Para isso, será necessário realizar os
seguintes passos, o primeiro seria a validação dos Modelos que foram utilizado
anteriormente e Comparar as previsões feitas por eles no caso o ARIMA e SARIMAX
(baseadas em dados de 2022 a setembro de 2024) com os dados atualizados até
31/12/2024, verificando a consistência dos modelos quanto à sazonalidade e picos de
demanda, o segundo passo seria a exploração de Novas Abordagens utilizando novos
modelos para que possam também integrar a sazonalidade comum e as tendências
individuais dos serviços, visando reduzir a taxa de erro das previsões, especialmente
para o serviço com comportamento volátil. 
    """
)

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

st.sidebar.success("Selecione um dashboard acima.")