import pandas as pd
from sklearn.cluster import KMeans

import plotly.express as px
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("export_IA_client1.csv")

# Création d'un cluster fixe "Stationnaire"
df["KMeans_Cluster"] = -1  # Valeur temporaire
df.loc[(df["SOG"] == 0) & (df["COG"] == 0), "KMeans_Cluster"] = 99  # Assignation du cluster 99 pour les navires immobiles

# Filtrage des navires non stationnaires
df_kmeans = df[df["KMeans_Cluster"] != 99].copy()

# Sélection des caractéristiques
features = ["SOG", "COG", "Heading"]
X = df_kmeans[features].values



# 🎯 Application de K-Means (4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
df_kmeans["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# Réintégration des résultats
df.update(df_kmeans)

# Sauvegarde des clusters
df.to_csv("clustering_results_with_stationnaire.csv", index=False)



# Chargement des données avec clustering
df = pd.read_csv("clustering_results_with_stationnaire.csv")

# Création du graphique
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df["SOG"], df["COG"], c=df["KMeans_Cluster"], cmap="tab10", alpha=0.7)

# Ajout de la légende et titres
plt.colorbar(scatter, label="Cluster")
plt.xlabel("SOG")
plt.ylabel("COG")
plt.title("Clustering des navires en fonction du cap et de la vitesse")
plt.grid(True)
plt.show()

fig = px.scatter_map(df, lat="LAT", lon="LON", color="KMeans_Cluster",
                     color_discrete_map={99: "red"},  # Navires stationnaires en rouge
                     map_style="open-street-map", zoom=5)

fig.show()
