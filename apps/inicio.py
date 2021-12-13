import streamlit as st 
import pandas as pd 

def app():
	st.title("Bienvenido a IntelliBlue, machine learning")
	st.caption('Esta es una aplicación que implementa distintos algoritmos de machine learning. Por ejemplo, ');
	st.caption('1. Algoritmo Apriori  \n2. Métricas de distancia  \n3. Clustering jerárquico y particional  \n4. Clasificación por regresión logística  \n5. Árboles de decisión (Pronóstico y Clasificación)');
	st.caption('\n\n\nPara probar los distintos algoritmos, esta aplicación proporciona distintos conjuntos de datos');

	st.caption('\t 🎦 Plataforma de películas')
	if st.button('🎦 Películas'):
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/movies.csv'
		Datos=pd.read_csv(file, header=None)
		st.write(Datos)

	st.caption('\t 🛍️ Productos vendidos en tienda minorista')
	if st.button('🛍️ Tienda'):
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/store_data.csv'
		Datos=pd.read_csv(file, header=None)
		st.write(Datos)

	st.caption('\t 🏡 Casos de hipoteca')
	if st.button('🏡 Hipoteca'):
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/Hipoteca.csv'
		Datos=pd.read_csv(file)
		st.write(Datos)

	st.caption('\t 😷 Diagnóstico de cáncer')
	if st.button('😷 Cáncer'):
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/WDBCOriginal.csv'
		Datos=pd.read_csv(file)
		st.write(Datos)

	st.caption('\t 💉 Diágnostico de diabetes')
	if st.button('💉 Diabetes'):
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/diabetes.csv'
		Datos=pd.read_csv(file)
		st.write(Datos)
