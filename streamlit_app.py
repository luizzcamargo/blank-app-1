import streamlit as st
from transformers import pipeline

# Título e descrição
st.set_page_config(page_title="Resuma seu Texto Aqui", page_icon="🧠")
st.title("Resumo de Texto")
st.write("Insira um texto e obtenha um resumo automático usando um modelo do Hugging Face. (acebook/bart-large-cnn)")

# Campo de texto
text_input = st.text_area("Digite ou cole seu texto abaixo:", height=200, placeholder="Cole seu texto aqui...")

# Botão de resumo
if st.button("Gerar Resumo"):
    if text_input.strip():
        with st.spinner("Gerando resumo..."):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(text_input, max_length=300, min_length=30, do_sample=False)
            st.subheader("Resumo gerado:")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("Por favor, insira um texto para resumir.")
