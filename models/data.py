import holidays
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import TimeSeriesSplit

def get_data():
    br_holidays = holidays.Brazil(state='SP')

    df_2022 = pd.read_csv('comissões 2022.csv', sep=";", encoding='ISO-8859-1', skiprows=7)
    df_2023 = pd.read_csv('comissões 2023.csv', sep=";", encoding='ISO-8859-1', skiprows=7)
    df_2024 = pd.read_csv('Comissões 2024 Completo.csv', sep=";", encoding='ISO-8859-1', skiprows=7)
    df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)

    df = df.drop(index=[2182, 4237, 6668])
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Assistente','Consumo de Pacote','CPF', 'Desconto Cliente', 'Motivo de Desconto','Desconto Operadora', 'Custo operacional','Profissional','Cliente','Comissão para','Quem registrou a transação', 'Valor', 'Valor Base Comissão', '% Comissão', 'Valor Comissão','Data de Liberação da Comissão','Pagamento / Estorno','Pago em'])
    df['Atendimento/Venda'] = pd.to_datetime(df['Atendimento/Venda'], format='%d/%m/%Y')
    
    df_serie = df['Atendimento/Venda'].value_counts().sort_index()
    df_serie = pd.DataFrame({'Data': df_serie.index, 'Atendimento/Venda': df_serie.values})
    df_serie['Dia'] = df_serie['Data'].dt.day
    df_serie['Dia da Semana'] = df_serie['Data'].dt.dayofweek
    df_serie['Mês'] = df_serie['Data'].dt.month
    df_serie['trimestre'] = df_serie['Data'].dt.quarter
    df_serie['Year'] = df_serie['Data'].dt.year
    df_serie['Dia do Ano'] = df_serie['Data'].dt.dayofyear
    df_serie['Feriado Estadual'] = df_serie['Data'].apply(lambda x: 1 if x in br_holidays else 0)
    df_serie = df_serie.iloc[1:]

    features = ['Dia', 'Dia da Semana', 'Mês', 'trimestre', 'Year', 'Dia do Ano', 'Feriado Estadual']

    X = df_serie[features].astype(int)
    Y = df_serie['Atendimento/Venda'].astype(float)
    stl = STL(Y.values, period=365)
    res = stl.fit()
    Y_deseason = Y.values - res.seasonal 
    tscv = TimeSeriesSplit(n_splits=4)

    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    Y_train_d, Y_test_d = Y_deseason[train_idx], Y_deseason[test_idx]
    Y_train_raw, Y_test_raw = Y.values[train_idx], Y.values[test_idx]

    return {
        'X': X,
        'Y': Y,
        'df_serie': df_serie,
        'X_train': X_train,
        'test_idx': test_idx,
        'X_test': X_test,
        'Y_deseason': Y_deseason,
        'Y_train_d': Y_train_d,
        'Y_test_d': Y_test_d,
        'tscv': tscv,
        'res': res,
        'Y_test_raw': Y_test_raw
    }
