import streamlit as st

from PIL import Image

from multiapp import MultiApp
from apps import alApriori, alMetricas, alCluster, alRegresion, alArbol, inicio


st.set_page_config(page_title="IntelliBlue", page_icon=':brain:')

st.image('logo1.png', width=100)

st.sidebar.info(" 👩‍💻 MACHINE LEARNING ")

app=MultiApp()
app.add_app("Seleccione algoritmo...", inicio.app)
app.add_app("Algoritmo Apriori", alApriori.app)
app.add_app("Métricas de distancia", alMetricas.app)
app.add_app("Clustering", alCluster.app)
app.add_app("Clasificación con Regresión Logística", alRegresion.app)
app.add_app("Árboles de decisión", alArbol.app)
app.run()
