import pandas as pd
import joblib
import plotly.express as px

# Chargement données et modèle
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")
df = pd.read_csv("4cluster_client1_results.csv")

# Entrée utilisateur
lat = float(input("Latitude (20 à 31) : "))
lon = float(input("Longitude (-98 à -78) : "))
sog = float(input("Vitesse SOG : "))
cog = float(input("Cap COG : "))

# Vérification
if not (20 <= lat <= 31 and -98 <= lon <= -78):
    print("Erreur : en dehors du Golfe du Mexique.")
    exit()

# Prédiction
new_point = pd.DataFrame([[lat, lon, sog, cog]], columns=["LAT", "LON", "SOG", "COG"])
new_scaled = scaler.transform(new_point)
predicted_cluster = kmeans.predict(new_scaled)[0]
print(f"Le navire appartient au cluster {predicted_cluster}")

# Intégration à la carte
new_point["KMeans_Cluster"] = "NOUVEAU_NAVIRE"
df["KMeans_Cluster"] = df["KMeans_Cluster"].astype(str)
df_map = pd.concat([df[["LAT", "LON", "KMeans_Cluster"]], new_point], ignore_index=True)

# Affichage
fig = px.scatter_geo(
    df_map,
    lat="LAT",
    lon="LON",
    color="KMeans_Cluster",
    symbol="KMeans_Cluster",
    symbol_map={"NOUVEAU_NAVIRE": "x"},
    projection="natural earth",
    title="Clusters de navires dans le Golfe du Mexique"
)

# Ajuster la vue sur le Golfe du Mexique
fig.update_geos(
    resolution=50,
    lataxis_range=[19, 32],
    lonaxis_range=[-99, -77],
    showland=True,
    landcolor="rgb(217, 217, 217)",
    showcountries=True,
    countrycolor="black",
    showocean=True,
    oceancolor="lightblue"
)

fig.update_traces(marker=dict(size=7))
fig.show()
