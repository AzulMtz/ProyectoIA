import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.tree import plot_tree, export_text
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

def app():
	st.title("ÁRBOLES DE DECISIÓN")
	data=st.radio("Selecciona los datos con los que deseas trabajar:",
				 			('🎦 Películas', '🛍️ Tienda', '🏡 Hipoteca', '😷 Cáncer *', '💉 Diabetes', 'Otros'), index=3)
	st.caption('** Datos recomendados')

	matriz=st.sidebar.radio("Árbol de decisión", ('Pronóstico', 'Clasificación'))

	if data=='🎦 Películas':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/movies.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos, matriz)

	elif data=='🛍️ Tienda':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/store_data.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos, matriz)

	elif data=='🏡 Hipoteca':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/Hipoteca.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, matriz)

	elif data=='😷 Cáncer *':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/WDBCOriginal.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, matriz)

	elif data=='💉 Diabetes':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/diabetes.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, matriz)
	
	elif data=='Otros':
		st.caption('En caso de que prefieras subir datos, el archivo debe tener extensión .csv')
		file=st.file_uploader("Sube tu archivo csv aquí", type=['csv'])
		if file is not None:
			if st.button('Subir archivo'):
				Datos=pd.read_csv(file)
				algoritmo(Datos, matriz)

def algoritmo(self, matriz):
	with st.expander("Estructura de los datos"):
		st.dataframe(self)
		st.write('La dimensión de los datos es: ', self.shape)
		st.write("Estadísticas de los datos")
		st.write(self.describe())
		var=st.multiselect("Variable a evaluar: ", self.columns, default=None)
		listo=st.checkbox('Listo', value=False)

	if listo:
		with st.expander("Selección de características"):
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
				X=np.array(self[caract])
				Y=np.array(self[var])
				st.write("Variables independientes")
				st.write(X)
				st.write("Variable a clasificar")
				st.write(Y)
		
			if matriz=='Pronóstico':
				with st.expander("Aplicación del algoritmo"):
					X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
		                                                                               		test_size = 0.20, 
		                                                                                	random_state = 0,
		                                                                                	shuffle = True)
					st.write("Hiperparámetros:")
					depth=st.number_input("Profundidad máxima: ", value=10)
					leaf=st.number_input("Elementos en las hojas: ", value=1)
					split=st.number_input("Elementos para la divisón: ", value=2)
					PronosticoAD = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, min_samples_split=split)
					PronosticoAD.fit(X_train, Y_train)
					Y_Pronostico = PronosticoAD.predict(X_test)
					Valores = pd.DataFrame(Y_test, Y_Pronostico)
					st.write("Valores reales contra pronosticados")
					st.write(Valores)
					plt.figure(figsize=(20, 5))
					plt.plot(Y_test, color='green', marker='o', label='Y_test')
					plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
					plt.title('Gráfica comparando valor pronosticado y valor real')
					plt.grid(True)
					plt.legend()
					st.pyplot(plt)
					exactitud = r2_score(Y_test, Y_Pronostico)
					st.write("**La exactitud es de: **", exactitud)
					validacion=st.checkbox('Validar modelo', value=False)

				if validacion:
					with st.expander("Validación del modelo"):
						st.write('Criterio: \n', PronosticoAD.criterion)
						Importancia= pd.DataFrame({'Variable': list(self[caract]),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
						st.write(Importancia)
						st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
						st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
						st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   
						st.write('Score: %.4f' % r2_score(Y_test, Y_Pronostico))
					with st.expander("Conformación del modelo"):
						st.write("Árbol de decisión")
						plt.figure(figsize=(16,16))  
						plot_tree(PronosticoAD, feature_names = caract)
						st.pyplot(plt)
						Reporte = export_text(PronosticoAD, feature_names = caract)
						st.download_button("📩 Descargar reporte del árbol", Reporte)
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
								Resultado = PronosticoAD.predict(pronostico)
								st.write("**El resultado es **", Resultado[0])

			elif matriz=='Clasificación':
				with st.expander("Aplicación del algoritmo"):
					X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
		                                                                                test_size = 0.2, 
		                                                                                random_state = 0,
		                                                                                shuffle = True)
					depth=st.number_input("Profundidad máxima: ", value=10)
					leaf=st.number_input("Elementos en las hojas: ", value=1)
					split=st.number_input("Elementos para la divisón: ", value=2)
					ClasificacionAD = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, min_samples_split=split)
					ClasificacionAD.fit(X_train, Y_train)
					Y_Clasificacion = ClasificacionAD.predict(X_validation)
					Valores = pd.DataFrame(Y_validation, Y_Clasificacion)
					st.write("Valores reales contra clasificados")
					st.write(Valores)
					exactitud = ClasificacionAD.score(X_validation, Y_validation)
					st.write("**La exactitud es de: **", exactitud)
					validacion=st.checkbox('Validar modelo', value=False)

				if validacion:
					with st.expander("Validación del modelo"):
						Y_Clasificacion = ClasificacionAD.predict(X_validation)
						Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
		                                   					Y_Clasificacion, 
		                                  					rownames=['Real'], 
		                                  					colnames=['Clasificación']) 
						st.write("Matriz de clasificación")
						st.write(Matriz_Clasificacion)
						st.write("Reporte de clasificación")
						st.write('Criterio: \n', ClasificacionAD.criterion)
						Importancia= pd.DataFrame({'Variable': list(self[caract]),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
						st.write(Importancia)
						st.write("Exactitud", ClasificacionAD.score(X_validation, Y_validation))
						st.write(classification_report(Y_validation, Y_Clasificacion))
					with st.expander("Conformación del modelo"):
						st.write("Árbol de decisión")

						plt.figure(figsize=(16,16))  
						plot_tree(ClasificacionAD, feature_names = caract, class_names = Y_Clasificacion)
						st.pyplot(plt)
						Reporte = export_text(ClasificacionAD, feature_names = caract)
						st.download_button("📩 Descargar reporte del árbol", Reporte)
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
								Resultado = ClasificacionAD.predict(pronostico)
								st.write("**El resultado es **", Resultado[0])