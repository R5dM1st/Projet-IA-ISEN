import pandas as pd
import numpy as np
import joblib

# Chargement données
df = pd.read_csv("4cluster_client1_results.csv")

scaler = joblib.load("scaler.pkl")
model = joblib.load("kmeans_model.pkl")

# Choisir un navire au hasard
random_ship = df.sample(n=1).iloc[0]

# Extraire les caractéristiques
lat = random_ship["LAT"]
lon = random_ship["LON"]
sog = random_ship["SOG"]
cog = random_ship["COG"]

print(f"Modèle chargé avec {model.n_clusters} clusters.")
print(f"Navire choisi au hasard :")
print(f"Latitude : {lat}, Longitude : {lon}, SOG : {sog}, COG : {cog}")

# Préparer les données pour prédiction
X_input = np.array([[lat, lon, sog, cog]])
X_scaled = scaler.transform(X_input)

# Prédire le cluster
cluster_pred = model.predict(X_scaled)[0]
print(f"Le modèle prédit que ce navire appartient au cluster : {cluster_pred}")
