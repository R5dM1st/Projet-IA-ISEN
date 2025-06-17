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

# Sauvegarde des résultats
df.to_csv("clustering_results.csv", index=False)

# 🔍 Visualisation des clusters K-Means
plt.scatter(df["LON"], df["LAT"], c=df["KMeans_Cluster"], cmap="viridis")
plt.xlabel("LON")
plt.ylabel("LAT")
plt.title("K-Means Clustering")
plt.show()

fig = px.scatter_map(df, lat="LAT", lon="LON", color="KMeans_Cluster",
                     map_style="open-street-map", zoom=5)
fig.show()
