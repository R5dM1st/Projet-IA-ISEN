import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score
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
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

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
