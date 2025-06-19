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

# Calcul du score silhouette (très long, simulé une fois puis mis en commentaire)
#silhouette_avg = silhouette_score(X_scaled, df["KMeans_Cluster"])
#print(f"Score silhouette du clustering K-Means : {silhouette_avg:.4f}")
#Score silhouette du clustering K-Means : 0.4865

# Calcul des scores
ch_score = calinski_harabasz_score(X_scaled, df["KMeans_Cluster"])
db_score = davies_bouldin_score(X_scaled, df["KMeans_Cluster"])
print(f"KM Calinski-Harabasz Index : {ch_score:.4f}")
print(f"KM Davies-Bouldin Index : {db_score:.4f}")
ch_score1 = calinski_harabasz_score(X_scaled, df["MBKMeans_Cluster"])
db_score1 = davies_bouldin_score(X_scaled, df["MBKMeans_Cluster"])
print(f"MBKM Calinski-Harabasz Index : {ch_score1:.4f}")
print(f"MBKM Davies-Bouldin Index : {db_score1:.4f}")

# Méthode du coude et calcul des scores
inertia = []
KM_calinski_scores = []
KM_davies_scores = []
MBKM_calinski_scores = []
MBKM_davies_scores = []
range_clusters = range(2, 11)  # Tester entre 2 et 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    mbkmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42)
    kmeans.fit(X_scaled)
    mbkmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    KM_calinski_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))
    KM_davies_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))
    MBKM_calinski_scores.append(calinski_harabasz_score(X_scaled, mbkmeans.labels_))
    MBKM_davies_scores.append(davies_bouldin_score(X_scaled, mbkmeans.labels_))

# Affichage des scores
print(f"KM Calinski-Harabasz Index : {KM_calinski_scores}")
print(f"KM Davies-Bouldin Index : {KM_davies_scores}")
print(f"MBKM Calinski-Harabasz Index : {MBKM_calinski_scores}")
print(f"MBKM Davies-Bouldin Index : {MBKM_davies_scores}")

# Graphique de la méthode du coude
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker="o", linestyle="-")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertia")
plt.title("Méthode du coude pour déterminer le nombre optimal de clusters")
plt.show()

#------ K-Means -------#

# Graphique des scores KM Davies-Bouldin
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, KM_davies_scores, marker="o", linestyle="-", label="Davies-Bouldin")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score")
plt.title("Comparaison des scores Davies-Bouldin avec KMeans")
plt.legend()
plt.show()

# Graphique des scores KM Calinski-Harabasz
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, KM_calinski_scores, marker="o", linestyle="-", label="Calinski-Harabasz")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score")
plt.title("Comparaison des scores Calinski-Harabasz avec KMeans")
plt.legend()
plt.show()

#------ MiniBatchK-Means -------#

# Graphique des scores MBKM Davies-Bouldin
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, MBKM_davies_scores, marker="o", linestyle="-", label="Davies-Bouldin")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score")
plt.title("Comparaison des scores Davies-Bouldin avec MiniBatchKMeans")
plt.legend()
#plt.show()

# Graphique des scores MBKM Calinski-Harabasz
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, MBKM_calinski_scores, marker="o", linestyle="-", label="Calinski-Harabasz")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score")
plt.title("Comparaison des scores Calinski-Harabasz avec MiniBatchKMeans")
plt.legend()
#plt.show()

#------ Visualisation ------#

# Visualisation des clusters K-Means
plt.scatter(df["LON"], df["LAT"], c=df["KMeans_Cluster"], cmap="viridis")
plt.xlabel("LON")
plt.ylabel("LAT")
plt.title("K-Means Clustering")
plt.show()

# Visualisation des clusters MBK-Means
plt.scatter(df["LON"], df["LAT"], c=df["MBKMeans_Cluster"], cmap="viridis")
plt.xlabel("LON")
plt.ylabel("LAT")
plt.title("MiniBatchK-Means Clustering")
#plt.show()

fig = px.scatter_map(df, lat="LAT", lon="LON", color="KMeans_Cluster",
                     map_style="open-street-map", zoom=5)
fig.show()
fig = px.scatter_map(df, lat="LAT", lon="LON", color="MBKMeans_Cluster",
                     map_style="open-street-map", zoom=5)
#fig.show()

# Sauvegarde des résultats
df.to_csv("clustering_client1_results.csv", index=False)
