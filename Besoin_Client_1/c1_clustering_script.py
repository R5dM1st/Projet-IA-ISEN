import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Chargement des données pré-traitées
df = pd.read_csv("export_IA_client1.csv")

# Sélection des caractéristiques de clustering
features = ["LAT", "LON", "SOG", "COG"]
X = df[features].values #on simplifie le nom pour la suite

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🎯 K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42) #randomstate=42 garantit des résultats réplicables à chaque simu
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled) #clustering avec les 4 variables normalisées

# Calcul du score silhouette (très long, simulé une fois puis mis en commentaire)
#silhouette_avg = silhouette_score(X_scaled, df["KMeans_Cluster"])
#print(f"Score silhouette du clustering K-Means : {silhouette_avg:.4f}")
#Score silhouette du clustering K-Means : 0.4865

# Calcul des scores
ch_score = calinski_harabasz_score(X_scaled, df["KMeans_Cluster"])
db_score = davies_bouldin_score(X_scaled, df["KMeans_Cluster"])
print(f"Calinski-Harabasz Index : {ch_score:.4f}")
print(f"Davies-Bouldin Index : {db_score:.4f}")

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

# 🔍 Visualisation des clusters K-Means
plt.scatter(df["LON"], df["LAT"], c=df["KMeans_Cluster"], cmap="viridis")
plt.xlabel("LON")
plt.ylabel("LAT")
plt.title("K-Means Clustering")
plt.show()

fig = px.scatter_map(df, lat="LAT", lon="LON", color="KMeans_Cluster",
                     map_style="open-street-map", zoom=5)
fig.show()

# Sauvegarde des résultats
df.to_csv("clustering_client1_results.csv", index=False)
