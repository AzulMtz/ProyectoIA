import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def app():
	st.title("Clasificación con Regresión Logística")
	data=st.radio("Selecciona los datos con los que deseas trabajar:",
				 			('🎦 Películas', '🛍️ Tienda', '🏡 Hipoteca *', '😷 Cáncer *', '💉 Diabetes', 'Otros'), index=3)
	st.caption('** Datos recomendados')

	if data=='🎦 Películas':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/movies.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos)

	elif data=='🛍️ Tienda':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/store_data.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos)

	elif data=='🏡 Hipoteca *':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/Hipoteca.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos)

	elif data=='😷 Cáncer *':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/WDBCOriginal.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos)

	elif data=='💉 Diabetes':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/diabetes.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos)

	elif data=='Otros':
		st.caption('En caso de que prefieras subir datos, el archivo debe tener extensión .csv')
		file=st.file_uploader("Sube tu archivo csv aquí", type=['csv'])
		if file is not None:
			if st.button('Subir archivo'):
				Datos=pd.read_csv(file)
				algoritmo(Datos)

def algoritmo(self):
	with st.expander("Estructura de los datos"):
		st.dataframe(self)
		st.write('La dimensión de los datos es: ', self.shape)
		var=st.multiselect("Variable a evaluar: ", self.columns, default=None)
		listo=st.checkbox('Listo', value=False)
		if listo:
			st.write(self.groupby(var).size())
	with st.expander("Selección de características"):
		graf=st.checkbox("Evaluación visual", value=False)
		if graf:
			sns.pairplot(self, hue=var[0])
			st.write("Evaluación visual")
			st.pyplot(plt)
		plt.figure(figsize=(14,7))
		CorrHipoteca=self.corr(method='pearson')
		MatrizInf = np.triu(CorrHipoteca)
		sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizInf)
		st.write("Mapa de calor")
		st.pyplot(plt)
		caract=st.multiselect("Selecciona las características a utilizar: ", self.columns)
		continuar=st.checkbox('Continuar', value=False)

	if continuar:
		with st.expander("Definición de variables"):
			var2=st.text_input("Cadena a reemplazar por 0: ")
			var3=st.text_input("Cadena a reemplazar por 1: ")
			remplazar=st.checkbox('Remplazar', value=False)
			if remplazar:
				self=self.replace({var2: 0, var3: 1})
			X=np.array(self[caract])
			Y=np.array(self[var])
			st.write("Variables independientes")
			st.write(X)
			st.write("Variable a clasificar")
			st.write(Y)
		with st.expander("Aplicación del algoritmo"):
			X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
	                                                                                test_size = 0.30, 
	                                                                                random_state =1234,
	                                                                                shuffle = True)
			Clasificacion = linear_model.LogisticRegression()
			Clasificacion.fit(X_train, Y_train)
			Probabilidad = Clasificacion.predict_proba(X_validation)
			Predicciones = Clasificacion.predict(X_validation)
			exactitud = Clasificacion.score(X_validation, Y_validation)
			st.write("**La exactitud es de: **", exactitud)
			validacion=st.checkbox('Validar modelo', value=False)

		if validacion:
			with st.expander("Validación del modelo"):
				Y_Clasificacion = Clasificacion.predict(X_validation)
				Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
				                                   Y_Clasificacion, 
				                                   rownames=['Real'], 
				                                   colnames=['Clasificación']) 
				st.write("Matriz de clasificación")
				st.write(Matriz_Clasificacion)
				st.write("Exactitud: ", exactitud)
				st.write("Reporte de clasificación")
				st.write(classification_report(Y_validation, Y_Clasificacion))
			with st.expander("Nuevos pronósticos"):
				with st.form("Pronóstico"):
					i=0
					nuevosDatos= []
					while i<len(caract):
						dato = st.number_input(caract[i])
						nuevosDatos.insert(i, dato)
						i=i+1
					nuevoPronostico = st.form_submit_button("Calcular")
					if nuevoPronostico:
						i=0
						pronostico = pd.DataFrame({'i': [0]})
						while i < len(caract):
							pronostico.insert(i, str(caract[i]), nuevosDatos[i])
							i=i+1
						del(pronostico['i'])
						Resultado = Clasificacion.predict(pronostico)
						st.write("**El resultado es **", Resultado[0])

