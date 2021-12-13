import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as shc 

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D

def app():
	st.title("CLUSTERING")
	data=st.radio("Selecciona los datos con los que deseas trabajar:",
				 			('🎦 Películas', '🛍️ Tienda', '🏡 Hipoteca *', '😷 Cáncer *', '💉 Diabetes', 'Otros'), index=2)
	st.caption('** Datos recomendados')

	matriz=st.sidebar.radio("Clustering", ('Particional', 'Jerárquico'))

	matriz2=st.sidebar.radio("Métrica de distancia", ('Euclidiana', 'Chebyshev', 'Manhattan'))
	
	if data=='🎦 Películas':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/movies.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos, matriz, matriz2)

	elif data=='🛍️ Tienda':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/store_data.csv'
		Datos=pd.read_csv(file, header=None)
		algoritmo(Datos, matriz, matriz2)

	elif data=='🏡 Hipoteca *':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/Hipoteca.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, matriz, matriz2)

	elif data=='😷 Cáncer *':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/WDBCOriginal.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, matriz, matriz2)

	elif data=='💉 Diabetes':
		file = 'https://raw.githubusercontent.com/AzulMtz/DatosIA/main/diabetes.csv'
		Datos=pd.read_csv(file)
		algoritmo(Datos, matriz, matriz2)

	elif data=='Otros':
		st.caption('En caso de que prefieras subir datos, el archivo debe tener extensión .csv')
		file=st.file_uploader("Sube tu archivo csv aquí", type=['csv'])
		if file is not None:
			Datos=pd.read_csv(file)
			algoritmo(Datos, matriz, matriz2)

def algoritmo(self, matriz, matriz2):
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
		if matriz=='Jerárquico':
			with st.expander("Aplicación del algoritmo"):
				MatrizDatos=np.array(self[caract])
				estandarizar=StandardScaler()
				MEstandarizada=estandarizar.fit_transform(MatrizDatos)
				plt.figure(figsize=(10, 7))
				plt.title("Casos")
				plt.ylabel('Distancia')
				if matriz2=='Euclidiana':
					Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
					st.pyplot(plt)
					num=st.number_input("¿Cuántos clústers hay? ", value=1)
					MJerarquico = AgglomerativeClustering(n_clusters=num, linkage='complete', affinity='euclidean')
					MJerarquico.fit_predict(MEstandarizada)
					MJerarquico.labels_
				elif matriz2=='Chebyshev':
					Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='chebyshev'))
					st.pyplot(plt)
					num=st.number_input("¿Cuántos clústers hay? ", value=1)
					MJerarquico = AgglomerativeClustering(n_clusters=num, linkage='complete', affinity='chebyshev')
					MJerarquico.fit_predict(MEstandarizada)
					MJerarquico.labels_
				elif matriz2=='Manhattan':
					Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='cityblock'))
					st.pyplot(plt)
					num=st.number_input("¿Cuántos clústers hay? ", value=1)
					MJerarquico = AgglomerativeClustering(n_clusters=num, linkage='complete', affinity='cityblock')
					MJerarquico.fit_predict(MEstandarizada)
					MJerarquico.labels_
			with st.expander("Cluster"):
				self['clusterH'] = MJerarquico.labels_
				st.write(self.groupby(['clusterH'])['clusterH'].count()) 
				numCluster=st.number_input("Ver clúster número ", value=0)
				st.write("Ver clúster número ", numCluster)
				st.write(self[self.clusterH == numCluster])
			with st.expander("Centroides"):
				CentroidesH = self.groupby('clusterH').mean()
				st.write(CentroidesH)

		elif matriz=='Particional':
			with st.expander("Aplicación del algoritmo"):
				MatrizDatos=np.array(self[caract])
				estandarizar=StandardScaler()
				MEstandarizada=estandarizar.fit_transform(MatrizDatos)
				SSE = []
				for i in range(2, 12):
				    km = KMeans(n_clusters=i, random_state=0)
				    km.fit(MEstandarizada)
				    SSE.append(km.inertia_)

				#Se grafica SSE en función de k
				plt.figure(figsize=(10, 7))
				plt.plot(range(2, 12), SSE, marker='o')
				plt.xlabel('Cantidad de clusters *k*')
				plt.ylabel('SSE')
				plt.title('Método del codo')
				st.pyplot(plt)
				kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
				st.write(kl.elbow, "clústers")
				MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
				MParticional.predict(MEstandarizada)
				MParticional.labels_
			with st.expander("Cluster"):
				self['clusterP'] = MParticional.labels_
				st.write(self.groupby(['clusterP'])['clusterP'].count()) 
				numCluster=st.number_input("Ver clúster número ", value=0)
				st.write("Ver clúster número ", numCluster)
				st.write(self[self.clusterP == numCluster])
			with st.expander("Centroides"):
				CentroidesP = self.groupby('clusterP').mean()
				st.write(CentroidesP)
			with st.expander("Gráfica de los elementos y centroides"):
				plt.rcParams['figure.figsize'] = (10, 7)
				plt.style.use('ggplot')
				colores=['red', 'blue', 'green', 'yellow']
				asignar=[]
				for row in MParticional.labels_:
				    asignar.append(colores[row])

				fig = plt.figure()
				ax = Axes3D(fig)
				ax.scatter(MEstandarizada[:, 0], 
				           MEstandarizada[:, 1], 
				           MEstandarizada[:, 2], marker='o', c=asignar, s=60)
				ax.scatter(MParticional.cluster_centers_[:, 0], 
				           MParticional.cluster_centers_[:, 1], 
				           MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
				st.pyplot(plt)