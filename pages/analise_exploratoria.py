import streamlit as st
import pandas as pd
import statsmodels.tsa.stattools as ts
import numpy as np
# Importing necessary libraries for visualization
import plotly.express as px
import plotly.graph_objects as go

# Set the page configuration
st.set_page_config(layout="wide", page_title="Análise de Comissões 2022-2024")

st.title("Análise Exploratória Salão de Beleza")
st.markdown("---")

# Load the dataset
df_2022 = pd.read_csv('comissões 2022.csv', sep=";", encoding='ISO-8859-1', skiprows=7)
df_2023 = pd.read_csv('comissões 2023.csv', sep=";", encoding='ISO-8859-1', skiprows=7)
df_2024 = pd.read_csv('Comissões 2024 Completo.csv', sep=";", encoding='ISO-8859-1', skiprows=7)
df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)

# Pre-Processamento dos dados
df = df.drop(index=[2182, 4237, 6668])
df = df.reset_index(drop=True)

#Descartando colunas vazias e que não foram consideradas necessárias para esse projeto
df = df.drop(columns=['Assistente','Consumo de Pacote','CPF', 'Desconto Cliente', 'Motivo de Desconto',
              'Desconto Operadora', 'Custo operacional','Profissional','Cliente','Comissão para','Quem registrou a transação', 'Valor', 'Valor Base Comissão', '% Comissão', 'Valor Comissão','Data de Liberação da Comissão','Pagamento / Estorno','Pago em'])

#Transformando a coluna data em seus devido tipo no caso tipo datetime
df['Atendimento/Venda'] = pd.to_datetime(df['Atendimento/Venda'], format='%d/%m/%Y')

df['Ano'] = df['Atendimento/Venda'].dt.year
df['Mês'] = df['Atendimento/Venda'].dt.month
df['Dia'] = df['Atendimento/Venda'].dt.day
df_servicos = df.copy()

df_ano_2022 = df[df['Ano'] == 2022]
df_ano_2023 = df[df['Ano'] == 2023]
df_ano_2024 = df[df['Ano'] == 2024]

meses_nomes = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}

atendimentos_por_mes_2022 = df_ano_2022['Mês'].value_counts().sort_index()
atendimentos_por_mes_2023 = df_ano_2023['Mês'].value_counts().sort_index()
atendimentos_por_mes_2024 = df_ano_2024['Mês'].value_counts().sort_index()

atendimentos_por_mes_2022.index = atendimentos_por_mes_2022.index.map(meses_nomes)
atendimentos_por_mes_2023.index = atendimentos_por_mes_2023.index.map(meses_nomes)
atendimentos_por_mes_2024.index = atendimentos_por_mes_2024.index.map(meses_nomes)

# Distribuicao de atendimentos por mes em cada ano

# Construir 3 layouts diferentes para o grafico de distribuiçao de atendimentos por mes em cada ano
# 1. Grafico de barras 2022
# 2. Grafico de barras 2023
# 3. Grafico de barras 2024
# 4. Grafico de barras comparativo entre os anos

df_wide = pd.DataFrame({
    '2022': atendimentos_por_mes_2022.values,
    '2023': atendimentos_por_mes_2023.values,
    '2024': atendimentos_por_mes_2024.values
}, index=meses_nomes)


df_wide.reset_index(inplace=True)
df_wide.rename(columns={'index': 'Mês'}, inplace=True)

df_long = pd.melt(
    df_wide,
    id_vars=['Mês'],                 
    value_vars=['2022', '2023', '2024'], 
    var_name='Ano',                  
    value_name='Atendimentos'        
)

fig = px.bar(
    df_long,
    x='Mês',
    y='Atendimentos',
    color='Ano',
    barmode='group',
    title='Frequência de Atendimentos 2022 a 2024',
    labels={'Atendimentos': 'Total de Atendimentos', 'Mês': 'Mês do Ano'},
    text_auto=True
)

fig.update_layout(xaxis_tickangle=-45)

col1, col2 = st.columns(2)
st.plotly_chart(fig, use_container_width=True)

#col1.plotly_chart(fig, use_container_width=True)

# Grafico da serie temporal de atendimentos de 2022 a 2024
fig2 = px.line(
    df_long,
    x='Mês',
    y='Atendimentos',
    color='Ano',
    title='Frequência de Atendimentos 2022 a 2024',
    labels={'Atendimentos': 'Total de Atendimentos', 'Mês': 'Mês do Ano'},
    markers=True
)

st.plotly_chart(
    fig2,
    use_container_width=True)

# Grafico de barras horizontal com numero da frequencia de servicos para cada ano
contagens_2022 = df_ano_2022['Serviço/Produto/Pacote'].value_counts()
contagens_2023 = df_ano_2023['Serviço/Produto/Pacote'].value_counts()
contagens_2024 = df_ano_2024['Serviço/Produto/Pacote'].value_counts()

contagens_2022 = contagens_2022.sort_values(ascending=True)
contagens_2023 = contagens_2023.sort_values(ascending=True)
contagens_2024 = contagens_2024.sort_values(ascending=True)

df_2022 = contagens_2022.reset_index().rename(columns={'index': 'Serviço/Produto/Pacote', 'count': 'Frequência'}); df_2022['Ano'] = 2022
df_2023 = contagens_2023.reset_index().rename(columns={'index': 'Serviço/Produto/Pacote', 'count': 'Frequência'}); df_2023['Ano'] = 2023
df_2024 = contagens_2024.reset_index().rename(columns={'index': 'Serviço/Produto/Pacote', 'count': 'Frequência'}); df_2024['Ano'] = 2024
df_combinado = pd.concat([df_2022, df_2023, df_2024])
df_combinado = df_combinado.sort_values(by='Frequência', ascending=True)

todos_servicos = df_combinado['Serviço/Produto/Pacote'].unique()
ano_selecionado = st.radio(
    "Selecione o ano",
    options=[2022, 2023, 2024],
    index=2,
    horizontal=True,
    key='filtro_ano_servico_fixo'
)

df_filtrado = df_combinado[df_combinado['Ano'] == ano_selecionado]

fig3 = px.bar(
    df_filtrado,
    x='Frequência',
    y='Serviço/Produto/Pacote',
    orientation='h',
    title=f'Frequência de Serviços/Produtos/Pacotes em {ano_selecionado}',
    text_auto=True
)

st.plotly_chart(fig3, use_container_width=True)

# Grafico de barras com numero de servico manutencao unha de gel de 2022 a 2024

# Grafico de serie temporal com o desempenho dos modelos

# Grafico com as metricas dos modelos

# Grafico com valores de previsao

