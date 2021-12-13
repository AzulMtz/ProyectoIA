import streamlit as st 
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance

def app():
	st.title("MÉTRICAS DE DISTANCIA")
	data=st.radio("Selecciona los datos con los que deseas trabajar:",
				 			('🎦 Películas', '🛍️ Tienda', '🏡 Hipoteca *', '😷 Cáncer', '💉 Diabetes', 'Otros'), index=2)
	st.caption('** Datos recomendados')

	matriz=st.sidebar.radio("Matriz de distancia", ('Euclidiana', 'Chebyshev', 'Manhattan', 'Minkoswki'))

	if data=='🎦 Películas':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/movies.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos, matriz)

	elif data=='🛍️ Tienda':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/store_data.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos, matriz)

	elif data=='🏡 Hipoteca *':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/Hipoteca.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, matriz)

	elif data=='😷 Cáncer':
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
			Datos=pd.read_csv(file)
			algoritmo(Datos, matriz)

def algoritmo(self, matriz):
	if matriz=='Euclidiana':
		DstEuclidiana = cdist(self, self, metric='euclidean')
		with st.expander("Matriz de distancia"):
			MEuclidiana = st.dataframe(DstEuclidiana)
			st.write('La dimensión de los datos es: ', self.shape)
		with st.expander("Distancia entre dos objetos"):
			st.write("📍 **Obtener distancia entre dos objetos** ")
			obj1=st.number_input("Objeto 1: ", 0, value=100)
			obj2=st.number_input("Objeto 2: ", 0, len(self)-1, value=200)
			Objeto1 = self.iloc[obj1]
			Objeto2 = self.iloc[obj2]
			dstEuclidiana = distance.euclidean(Objeto1, Objeto2)
			st.write("La distancia entre el objeto ", obj1, "y el objeto ", obj2, "es: ", dstEuclidiana)
		with st.expander("Distancia en un rango de datos"):
			st.write("📍 **Obtener distancia en un rango de datos** ")
			obj3=st.number_input("Rango inferior 1: ", 0, value=0)
			obj4=st.number_input("Rango superior 1: ", 0, value=10)
			obj5=st.number_input("Rango inferior 2: ", 0, value=0)
			obj6=st.number_input("Rango superior 2: ", 0, value=10)
			DstEuclidiana = cdist(self.iloc[obj3:obj4], self.iloc[obj5:obj6], metric='euclidean')
			MEuclidiana = st.dataframe(DstEuclidiana)
		
	elif matriz=='Chebyshev':
		DstChebyshev = cdist (self, self, metric='chebyshev')
		with st.expander("Matriz de distancia"):
			MChebyshev = st.dataframe(DstChebyshev)
			st.write('La dimensión de los datos es: ', self.shape)
		with st.expander("Distancia entre dos objetos"):
			st.write("📍 **Obtener distancia entre dos objetos** ")
			obj1=st.number_input("Objeto 1: ", 0, value=100)
			obj2=st.number_input("Objeto 2: ", 0, len(self)-1, value=200)
			Objeto1 = self.iloc[obj1]
			Objeto2 = self.iloc[obj2]
			dstChebyshev = distance.chebyshev(Objeto1, Objeto2)
			st.write("La distancia entre el objeto ", obj1, "y el objeto ", obj2, "es: ", dstChebyshev)
		with st.expander("Distancia en un rango de datos"):
			st.write("📍 **Obtener distancia en un rango de datos** ")
			obj3=st.number_input("Rango inferior 1: ", 0, value=0)
			obj4=st.number_input("Rango superior 1: ", 0, value=10)
			obj5=st.number_input("Rango inferior 2: ", 0, value=0)
			obj6=st.number_input("Rango superior 2: ", 0, value=10)
			DstChebyshev = cdist(self.iloc[obj3:obj4], self.iloc[obj5:obj6], metric='chebyshev')
			MChebyshev = st.dataframe(DstChebyshev)

	elif matriz=='Manhattan':
		DstManhattan = cdist (self, self, metric='cityblock')
		with st.expander("Matriz de distancia"):
			MManhattan = st.dataframe(DstManhattan)
			st.write('La dimensión de los datos es: ', self.shape)
		with st.expander("Distancia entre dos objetos"):
			st.write("📍 **Obtener distancia entre dos objetos** ")
			obj1=st.number_input("Objeto 1: ", 0, value=100)
			obj2=st.number_input("Objeto 2: ", 0, len(self)-1, value=200)
			Objeto1 = self.iloc[obj1]
			Objeto2 = self.iloc[obj2]
			dstManhattan = distance.cityblock(Objeto1, Objeto2)
			st.write("La distancia entre el objeto ", obj1, "y el objeto ", obj2, "es: ", dstManhattan)
		with st.expander("Distancia en un rango de datos"):
			st.write("📍 **Obtener distancia en un rango de datos** ")
			obj3=st.number_input("Rango inferior 1: ", 0, value=0)
			obj4=st.number_input("Rango superior 1: ", 0, value=10)
			obj5=st.number_input("Rango inferior 2: ", 0, value=0)
			obj6=st.number_input("Rango superior 2: ", 0, value=10)
			DstManhattan = cdist(self.iloc[obj3:obj4], self.iloc[obj5:obj6], metric='cityblock')
			MManhattan = st.dataframe(DstManhattan)

	elif matriz=='Minkoswki':
		DstMinkowski = cdist (self, self, metric='minkowski', p=1.5)
		with st.expander("Matriz de distancia"):
			MMinkowski = st.dataframe(DstMinkowski)
			st.write('La dimensión de los datos es: ', self.shape)
		with st.expander("Distancia entre dos objetos"):
			st.write("📍 **Obtener distancia entre dos objetos** ")
			obj1=st.number_input("Objeto 1: ", 0, value=100)
			obj2=st.number_input("Objeto 2: ", 0, len(self)-1, value=200)
			Objeto1 = self.iloc[obj1]
			Objeto2 = self.iloc[obj2]
			dstMinkowski = distance.minkowski(Objeto1, Objeto2, 1.5)
			st.write("La distancia entre el objeto ", obj1, "y el objeto ", obj2, "es: ", dstMinkowski)
		with st.expander("Distancia en un rango de datos"):
			st.write("📍 **Obtener distancia en un rango de datos** ")
			obj3=st.number_input("Rango inferior 1: ", 0, value=0)
			obj4=st.number_input("Rango superior 1: ", 0, value=10)
			obj5=st.number_input("Rango inferior 2: ", 0, value=0)
			obj6=st.number_input("Rango superior 2: ", 0, value=10)
			DstMinkowski = cdist(self.iloc[obj3:obj4], self.iloc[obj5:obj6], metric='minkowski')
			MMinkowski = st.dataframe(DstMinkowski)
