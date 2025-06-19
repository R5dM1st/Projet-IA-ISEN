import joblib
import numpy as np

# Chargement du modèle et du scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("kmeans_model.pkl")  # ou "mbkmeans_model.pkl"

# Saisie utilisateur
try:
    lat = float(input("Entrez la LATITUDE (LAT) : "))
    lon = float(input("Entrez la LONGITUDE (LON) : "))
    sog = float(input("Entrez la VITESSE (SOG) : "))
    cog = float(input("Entrez le CAP (COG) : "))

    # Prédiction
    input_data = np.array([[lat, lon, sog, cog]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)

    print(f"\n✅ Le navire appartient au cluster : {cluster[0]}")

except ValueError:
    print("❌ Erreur : veuillez entrer des valeurs numériques valides.")
