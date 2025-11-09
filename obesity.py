# Importação de bibliotecas

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregamento do pipeline usando decorador

NOME_PIPELINE = 'modelo_obesidade_pipeline_COMPLETO.joblib'

@st.cache_data
def carrega_pipeline():
    """Carrega o Pipeline de ML salvo com joblib."""
    try:
        pipeline = joblib.load(NOME_PIPELINE)
        return pipeline
    except FileNotFoundError:
        st.error(f"Erro: O arquivo do pipeline '{NOME_PIPELINE}' não foi encontrado.")
        return None

pipeline_modelo = carrega_pipeline()


# Funções para a formatação dos dados categóricos obtidos via Streamlit para o formato esperado pelo pipeline

def converte_binarios(df):

    binarios = ['historico_familiar','come_alimentos_caloricos','fuma','monitora_calorias_consumidas']

    for coluna in binarios:
        df[coluna] = df[coluna].apply(lambda x: 1 if x == 'Sim' else 0)

    print(f'Campos {binarios} convertidos para binário. Sim = 1, Não = 0')
    return df

def converte_genero(df):

    df['genero'] = df['genero'].apply(lambda x: 1 if x == 'Mulher' else 0)
    print('Campo de genero convertido para binário. Mulher = 1, Homem = 0')
    return df

def converte_categoricos (df):

    categoricos = ['come_entre_refeicoes', 'consome_alcool']

    dict = {    'Não' : 0,
                'Algumas vezes' : 1,
                'Frequentemente' : 2,
                'Sempre' : 3
}

    for col in categoricos:
        df[col] = df[col].map(dict)

    print(f'Campos categóricos {categoricos} convertidos.')
    return df

def converte_costuma_comer_vegetais (df):

    categoricos = ['costuma_comer_vegetais']

    dict = {    'Raramente' : 0,
                'Às vezes' : 1,
                'Sempre' : 2
}

    for col in categoricos:
        df[col] = df[col].map(dict)

    print(f'Campos categóricos (vegetais) {categoricos} convertidos.')
    return df


# Configuração da Página
st.set_page_config(
    page_title="Preditor de Obesidade", # Título que aparece na aba do navegador
    page_icon="⚖️",                      # Ícone na aba do navegador (pode ser um emoji ou caminho para um arquivo)
    layout="wide",                       # Define o layout para ocupar toda a largura da tela
    initial_sidebar_state="auto"         # Define o estado inicial da barra lateral
)

# Título Principal
st.title("⚖️ Preditor de Nível de Obesidade")

# Descrição/Subtítulo
st.markdown(
    """
    Este aplicativo utiliza um modelo de Machine Learning (Random Forest) 
    para prever o nível de obesidade de um indivíduo com base em dados antropométricos e hábitos de vida.
    **Preencha os campos abaixo para obter a previsão:**
    """
)

st.divider() # Linha final do cabeçalho

# Captura das Features

col1, col2, col3 = st.columns(3)

# Coluna 1

with col1:

    st.subheader("Dados Básicos")

    idade = st.number_input("Idade (anos)", min_value=18, max_value=120, value=25, step=1)

    altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")

    peso = st.number_input("Peso (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1, format="%.1f") 

    genero = st.radio("Gênero", options=['Mulher', 'Homem'], horizontal=True, index=0)

    historico_familiar = st.radio("Histórico Familiar de Sobrepeso/Obesidade", options=['Sim', 'Não'], horizontal=True, index=1)

# Coluna 2
with col2:
    st.subheader("Hábitos Alimentares")

    numero_refeicoes_diarias = st.number_input("Número de Refeições Diárias", min_value=1, max_value=10, value=3, step=1)

    consumo_diario_agua = st.number_input("Consumo Diário de Água (litros)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.1f")

    come_alimentos_caloricos = st.radio("Costuma Comer Alimentos Calóricos", options=['Sim', 'Não'], horizontal=True, index=1)  

    costuma_comer_vegetais = st.radio("Costuma Comer Vegetais", options=['Raramente', 'Às vezes', 'Sempre'], horizontal=True, index=0)

    come_entre_refeicoes = st.radio("Costuma Comer Entre as Refeições", options=['Não', 'Às vezes', 'Frequentemente', 'Sempre'], horizontal=True, index=1)

    monitora_calorias_consumidas = st.radio("Monitora as Calorias Consumidas", options=['Sim', 'Não'], horizontal=True, index=1)


# Coluna 3
with col3:
    st.subheader("Outros Dados")

    frequencia_atividade_fisica = st.number_input("Frequência de Atividade Física (dias por semana)", min_value=0, max_value=7, value=3, step=1)

    tempo_dispositivos_tecnologicos = st.number_input("Tempo em Dispositivos Tecnológicos (horas por dia)", min_value=0, max_value=24, value=4, step=1)

    meio_de_transporte = st.selectbox("Meio de Transporte Principal", options=['Transporte_Publico', 'Caminhando', 'Automovel', 'Motocicleta', 'Bicicleta'], index=0)

    col3_1, col3_2 = st.columns(2)
    with col3_1:
        consome_alcool = st.radio("Consome Álcool", options=['Não', 'Às vezes', 'Frequentemente', 'Sempre'], horizontal=True, index=1)
    with col3_2:
        fuma = st.radio("Fuma", options=['Sim', 'Não'], horizontal=True, index=1)



st.divider() # Linha separadora antes do botão

# Botão de Previsão
if st.button("Prever Nível de Obesidade"):
    if pipeline_modelo is not None:
        # Criação do DataFrame com os dados de entrada
        dados_entrada = pd.DataFrame({
            'genero': [genero],
            'idade': [idade],
            'altura': [altura],
            'peso': [peso],
            'historico_familiar': [historico_familiar],
            'come_alimentos_caloricos': [come_alimentos_caloricos],
            'costuma_comer_vegetais': [costuma_comer_vegetais],
            'numero_refeicoes_diarias': [numero_refeicoes_diarias],
            'come_entre_refeicoes': [come_entre_refeicoes],
            'fuma': [fuma],
            'consumo_diario_agua': [consumo_diario_agua],
            'monitora_calorias_consumidas': [monitora_calorias_consumidas],
            'frequencia_atividade_fisica': [frequencia_atividade_fisica],
            'tempo_dispositivos_tecnologicos': [tempo_dispositivos_tecnologicos],
            'consome_alcool': [consome_alcool],
            'meio_de_transporte': [meio_de_transporte]
        })

        # Formatação dos dados categóricos
        dados_entrada = converte_binarios(dados_entrada)
        dados_entrada = converte_genero(dados_entrada)
        dados_entrada = converte_categoricos(dados_entrada)
        dados_entrada = converte_costuma_comer_vegetais(dados_entrada)

        # Realiza a previsão usando o pipeline carregado
        previsao = pipeline_modelo.predict(dados_entrada)

        # Mapeamento da previsão para rótulos legíveis
        mapa_obesidade = {
            0: "Abaixo do Peso",
            1: "Peso Normal",
            2: "Sobrepeso I",
            3: "Sobrepeso II",
            4: "Obesidade Grau I",
            5: "Obesidade Grau II",
            6: "Obesidade Grau III"
        }

        nivel_obesidade = mapa_obesidade.get(previsao[0], "Desconhecido")

        # Exibe o resultado da previsão
        st.success(f"O nível de obesidade previsto é: **{nivel_obesidade}**")
    else:
        st.error("O modelo não está disponível no momento. Tente novamente mais tarde.")