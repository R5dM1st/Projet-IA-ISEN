#partie client 1

import pandas as pd

# Chargement des données brutes
df = pd.read_csv("vessel-clean-final.csv")
columns_to_keep = ["BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]
df_selected = df[columns_to_keep]
#pas besoin d'encoder de valeurs ici donc on exporte
df_selected.to_csv("export_IA_client1.csv", index=False)
