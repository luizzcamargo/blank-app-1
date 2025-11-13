import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Resuma seu Texto Aqui", page_icon="🧠")
st.title("Resumo de Texto")
st.write("Insira um texto e obtenha um resumo automático usando um modelo do Hugging Face (facebook/bart-large-cnn).")

# Carrega o modelo apenas uma vez — MUITO mais rápido
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Entrada de texto
text_input = st.text_area("Digite ou cole seu texto abaixo:", height=250, placeholder="Cole seu texto aqui...")

# Configurações ajustáveis (opcional)
st.sidebar.header("Configurações do Resumo")
max_len = st.sidebar.slider("Tamanho máximo do resumo", 100, 1024, 400)
min_len = st.sidebar.slider("Tamanho mínimo do resumo", 20, 200, 80)
beam_size = st.sidebar.slider("Qualidade (num_beams)", 2, 10, 4)

if st.button("Gerar Resumo"):
    if text_input.strip():
        with st.spinner("Gerando resumo..."):
            summary = summarizer(
                text_input,
                max_length=max_len,
                min_length=min_len,
                num_beams=beam_size,
                do_sample=False
            )

            st.subheader("Resumo gerado:")
            st.success(summary[0]['summary_text'])
    
    else:
        st.warning("Por favor, insira um texto para resumir.")

