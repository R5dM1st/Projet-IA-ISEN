import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Chargement des données
df = pd.read_csv("export_IA_client1.csv")

# Sélection des caractéristiques
features = ["LAT", "LON", "SOG", "COG"]
X = df[features].values

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 🎯 K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42) #randomstate=42 garantit des résultats réplicables à chaque simu
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled) #clustering avec les 4 variables normalisées

# Calcul du score silhouette (très long, simulé une fois puis mis en commentaire)
silhouette_avg = silhouette_score(X_scaled, df["KMeans_Cluster"])
print(f"Score silhouette du clustering K-Means : {silhouette_avg:.4f}")