import streamlit as st
import pdfminer
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configuración de la app
st.title("Ley de Guatemala")
st.subheader("Análisis de artículos")

# Cargar el modelo
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)

# Cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Función para extraer el contenido de cada artículo
def extraer_contenido(texto):
    articulos = []
    for articulo in texto.split("Artículo "):
        articulo_numero = articulo.split(".")[0]
        articulo_contenido = articulo.split(".")[1:]
        articulo_contenido = " ".join(articulo_contenido)
        articulos.append((articulo_numero, articulo_contenido))
    return articulos

# Función para analizar el contenido de cada artículo
def analizar_contenido(articulos):
    resultados = []
    for articulo in articulos:
        input_ids = tokenizer.encode(articulo[1], return_tensors="pt")
        attention_mask = tokenizer.encode(articulo[1], return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        outputs = model(input_ids, attention_mask=attention_mask)
        resultado = outputs.logits.argmax(-1).item()
        resultados.append((articulo[0], resultado))
    return resultados

# Función para mostrar los resultados
def mostrar_resultados(resultados):
    st.write("Resultados:")
    for resultado in resultados:
        st.write(f"Artículo {resultado[0]}: {resultado[1]}")

# Cargar el archivo PDF
st.file_uploader("Cargar archivo PDF", type=["pdf"])

# Procesar el archivo PDF
if st.button("Procesar"):
    archivo_pdf = st.file_uploader("Cargar archivo PDF", type=["pdf"])
    texto = pdfminer.extract_text(archivo_pdf)
    articulos = extraer_contenido(texto)
    resultados = analizar_contenido(articulos)
    mostrar_resultados(resultados)
