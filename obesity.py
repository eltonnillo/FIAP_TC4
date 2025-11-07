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
    st.subheader = ("Informações Básicas")

    genero = st.radio("Gênero", options=['Mulher', 'Homem'], horizontal=True, index=0)
