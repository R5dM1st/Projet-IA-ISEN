import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Charger scaler et modèle
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# Charger données
df = pd.read_csv("4cluster_client1_results.csv")

# Entrée utilisateur (exemple simplifié)
lat = float(input("Latitude (20 à 31) : "))
lon = float(input("Longitude (-98 à -78) : "))
sog = float(input("Vitesse SOG : "))
cog = float(input("Cap COG : "))

# Vérification zone
if not (20 <= lat <= 31 and -98 <= lon <= -78):
    print("Erreur : hors zone Golfe du Mexique")
    exit()

# Préparer nouveau navire
new_point = pd.DataFrame([[lat, lon, sog, cog]], columns=["LAT", "LON", "SOG", "COG"])
new_scaled = scaler.transform(new_point.values)
predicted_cluster = kmeans.predict(new_scaled)[0]
print(f"Le navire appartient au cluster {predicted_cluster}")

# Carte des clusters
fig = px.scatter_mapbox(
    df,
    lat="LAT",
    lon="LON",
    color=df["KMeans_Cluster"].astype(str),
    zoom=5,
    height=600,
    mapbox_style="open-street-map",
    title="Clusters de navires avec nouveau navire"
)

# Ajouter nouveau navire avec un gros symbole croix rouge
fig.add_trace(go.Scattermapbox(
    lat=[lat],
    lon=[lon],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=20,
        color='red',
        opacity=0.8,
        symbol='star'  # optionnel, pour changer la forme
    ),
    name='Nouveau navire'
))
fig.show()
