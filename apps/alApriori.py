import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
from apyori import apriori 

def app():

	st.title("ALGORITMO APRIORI")

	data=st.radio("Selecciona los datos con los que deseas trabajar:",
				 			('🎦 Películas *', '🛍️ Tienda *', '🏡 Hipoteca', '😷 Cáncer', '💉 Diabetes', 'Otros'))
	st.caption('** Datos recomendados')
	support=st.sidebar.slider('Soporte: ', 0.00, 1.00, 0.01)
	confidence=st.sidebar.slider('Confianza: ', 0.00, 1.00, 0.3)
	lift=st.sidebar.slider('Elevación: ', 0, 100, 2)

	if data=='🎦 Películas *':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/movies.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos, support, confidence, lift)

	elif data=='🛍️ Tienda *':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/store_data.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos, support, confidence, lift)

	elif data=='🏡 Hipoteca':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/Hipoteca.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, support, confidence, lift)

	elif data=='😷 Cáncer':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/WDBCOriginal.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, support, confidence, lift)

	elif data=='💉 Diabetes':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/diabetes.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, support, confidence, lift)
	
	elif data=='Otros':
		st.caption('En caso de que prefieras subir datos, el archivo debe tener extensión .csv')
		file=st.file_uploader("Sube tu archivo csv aquí", type=['csv'])
		if file is not None:
			Datos=pd.read_csv(file)
			algoritmo(Datos, support, confidence, lift)

def algoritmo(self, support, confidence, lift):
	with st.expander("Estructura de los datos"):
		st.dataframe(self)
		st.write('La dimensión de los datos es: ', self.shape)
	with st.expander("Elementos ordenados por su frecuencia"):
		Tran = self.values.reshape(-1).tolist()
		Lista = pd.DataFrame(Tran)
		Lista['Frecuencia'] = 0;
		Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo y ordenamiento
		Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
		Lista = Lista.rename(columns={0 : 'Item'})
		st.write(Lista)
	with st.expander("Gráfica Elemento-Frecuencia"):
		plt.figure(figsize=(16,20), dpi=300)
		plt.ylabel('Item')
		plt.xlabel('Frecuencia')
		plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
		st.pyplot(plt)
	DatosLista = self.stack().groupby(level=0).apply(list).tolist()
	ReglasC=apriori(DatosLista, min_support=support, min_confidence=confidence, min_lift=lift)
	Resultados=list(ReglasC)
	with st.expander("Reglas de asociación"):
		st.write('Se encontraron: ', len(Resultados), ' reglas.')
		for item in Resultados:
		  #El primer índice de la lista
		  Emparejar = item[0]
		  items = [x for x in Emparejar]
		  st.write("Regla: " + str(item[0]))

		  #El segundo índice de la lista
		  st.write("Soporte: " + str(item[1]))

		  #El tercer índice de la lista
		  st.write("Confianza: " + str(item[2][0][2]))
		  st.write("Lift: " + str(item[2][0][3])) 
		  st.write("=====================================") 