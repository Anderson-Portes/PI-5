import streamlit as st

st.set_page_config(
    page_title="Aprimoramento e Validação de Modelos para Séries Temporais",
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
