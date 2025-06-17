import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Chargement des données
df = pd.read_csv("export_IA_client1.csv")

# Sélection des caractéristiques
features = ["LAT", "LON", "SOG", "COG"]
X = df[features].values

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🎯 Méthode du coude et calcul des scores
inertia = []
calinski_scores = []
davies_scores = []
range_clusters = range(2, 11)  # Tester entre 2 et 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    calinski_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))
    davies_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))

# 🏆 Affichage des scores
print(f"Calinski-Harabasz Index : {calinski_scores}")
print(f"Davies-Bouldin Index : {davies_scores}")

# 🔍 Graphique de la méthode du coude
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker="o", linestyle="-")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertia")
plt.title("Méthode du coude pour déterminer le nombre optimal de clusters")
plt.show()

# 🔍 Graphique des scores Davies-Bouldin
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, davies_scores, marker="o", linestyle="-", label="Davies-Bouldin")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score")
plt.title("Comparaison des scores Davies-Bouldin")
plt.legend()
plt.show()

# 🔍 Graphique des scores Calinski-Harabasz
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, calinski_scores, marker="o", linestyle="-", label="Calinski-Harabasz")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score")
plt.title("Comparaison des scores Calinski-Harabasz")
plt.legend()
plt.show()
