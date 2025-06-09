import streamlit as st

st.set_page_config(page_title="Conclusão",layout="wide")

st.header("Conclusão")
st.write(
    """
Levando em conta a comparação com os modelos SARIMAX e ARIMA
aplicados anteriormente às previsões mensais de atendimentos totais e de manutenção
de unha de gel, constatamos que a mudança para projeções diárias trouxe ganhos
substanciais em flexibilidade e precisão. Ao aumentar o número de observações na
série, conseguimos explorar melhor a variabilidade cotidiana dos atendimentos, ao
mesmo tempo em que a dessazonalização prévia garantiu que padrões repetitivos não
viessem a enviesar as estimativas.
A adoção de cinco abordagens distintas ARIMA, SARIMAX, Prophet,
XGBoost e modelos bayesianos, integrada a validações cruzadas rigorosas, permitiu
selecionar soluções robustas tanto para o volume total de atendimentos quanto para
serviços individuais de comportamento mais volátil. O Prophet destacou-se pela sua
habilidade nativa de capturar sazonalidades múltiplas e mudanças de tendência,
enquanto o framework bayesiano ofereceu uma visão complementar sobre a incerteza
inerente a cada parâmetro.
Em síntese, a combinação de uma granularidade diária com técnicas avançadas
de modelagem e avaliação elevou a confiabilidade das previsões, o que pode fornecer à gestão do salão de beleza insights mais detalhados para o planejamento de equipe,
estoque de insumos e campanhas promocionais.
"""
)

