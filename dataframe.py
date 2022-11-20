#Importation des données
import pandas as pd

df = pd.read_table("market_basket.txt",delimiter="\t",header=0)

#Affichage des 10 premières lignes
print(df.head(10))