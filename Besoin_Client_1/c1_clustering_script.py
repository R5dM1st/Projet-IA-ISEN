import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

# Chargement des données pré-traitées
df = pd.read_csv("export_IA_client1.csv")

# Sélection des caractéristiques de clustering
features = ["SOG", "COG", "Heading"]
X = df[features].values #on simplifie le nom pour la suite

# 🎯 K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42) #randomstate=42 garantit des résultats réplicables à chaque simu

df["KMeans_Cluster"] = kmeans.fit_predict(X)

# 🎯 DBSCAN Clustering
#dbscan = DBSCAN(eps=0.5, min_samples=10)
#df["DBSCAN_Cluster"] = dbscan.fit_predict(X)

# Sauvegarde des résultats
df.to_csv("clustering_results.csv", index=False)

# 🔍 Visualisation des clusters K-Means
plt.scatter(df["SOG"], df["COG"], c=df["KMeans_Cluster"], cmap="Dark2")
plt.xlabel("SOG (Speed)")
plt.ylabel("COG (Course)")
plt.title("K-Means Clustering")
plt.show()
