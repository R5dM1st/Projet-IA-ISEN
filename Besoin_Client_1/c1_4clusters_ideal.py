import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import joblib

# Chargement des données pré-traitées
df = pd.read_csv("export_IA_client1.csv")

# Sélection des caractéristiques de clustering
features = ["LAT", "LON", "SOG", "COG"]
X = df[features].values #on simplifie le nom pour la suite

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42) #randomstate=42 garantit des résultats réplicables à chaque simu
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled) #clustering avec les 4 variables normalisées

# MiniBatchKMeans
mbkmeans = MiniBatchKMeans(n_clusters=4, batch_size=10000, random_state=42)
df["MBKMeans_Cluster"] = mbkmeans.fit_predict(X_scaled)

# Sauvegarde des résultats
df.to_csv("4cluster_client1_results.csv", index=False)

# Sauvegarde du scaler et du modèle
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
