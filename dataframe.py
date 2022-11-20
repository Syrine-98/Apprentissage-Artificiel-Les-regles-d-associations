#Importation de la fonction "apriori"
from mlxtend.frequent_patterns import apriori
#Importation de numpy
import numpy 
#Importation de la fonction de calcul des règles
from mlxtend.frequent_patterns import association_rules
#Importation des données
import pandas as pd

df = pd.read_table("market_basket.txt",delimiter="\t",header=0)

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
#Affichage des 10 premières lignes
print("\n")
print("Les 10 premières lignes du DataFrame : ")
print(df.head(10))

#Affichage des dimensions du Dataframe
print("\n")
print("Les dimensions du Dataframe sont : ")
print(df.shape)

#La fonction Construire()
def construire():
    crosstable = pd.DataFrame(index=range(int(df["ID"].iloc[-1:])))
    crosstable["ID"] = range(1,int(df["ID"].iloc[-1:]+1))
    for i in df.index:
        prod = df["Product"][i]
        crosstable[prod] = 0
        for j in crosstable.index:
            if (df["ID"][i] == crosstable["ID"][j]):
                crosstable[prod][j] = 1
    return (crosstable.head(10))
print("\n")
print("Le résultat de la fonction Construire() est :")
print(construire())

#Utilisation de la bibliothèque pandas.crosstab 
CrossTab= pd.crosstab(df.ID,df.Product)
#Affichage de la présence des produits au niveau des caddies 
print("\n")
print("Affichage de la présence des produits au niveau des caddies : ")
print(CrossTab)

#Affichage des 30 premières transactions et des 3 premiers produits.
print("\n")
print("Affichage des 30 premières transactions et des 3 premiers produits : ")
print(CrossTab.iloc[:30,:3])

#Extraction des itemsets les plus fréquents : min_supp=0.025 et un longueur maximum de 4 produits)
print("\n")
itemsets_freq = apriori(CrossTab,min_support=0.025,max_len=4,use_colnames=True)

#Affichage des dimensions d'itemsets fréquents
print("\n")
print("Les dimensions pour les itemsets fréquents sont : ")
print(itemsets_freq.shape)

#Affichage des 15 premiers itemsets fréquents
print("\n")
print("Les 15 premiers itemsets fréquents sont : ")
print(itemsets_freq.head(15))

#is_inclus() : une fonction qui permet de vérifier si un sous-ensemble items est inclus dans l’ensemble x
def is_inclus(x,items):
    return items.issubset(x)

#Affichage des itemsets comprenant le produit 'Aspirin' en utilisant la fonction "is_inclus()"
R1 = numpy.where(itemsets_freq.itemsets.apply(is_inclus,items={'Aspirin'}))
print("\n")
print("Les itemsets comprenant le produit 'Aspirin' sont : ")
print(itemsets_freq.loc[R1])

#Affichage des itemsets contenant les produits "Aspirin" et "Eggs"
R2 = numpy.where(itemsets_freq.itemsets.apply(is_inclus,items={'Aspirin','Eggs'}))
print("\n")
print("Les itemsets contenant les produits 'Aspirin' et 'Eggs' sont : ")
print(itemsets_freq.loc[R2])

#Production des règles à partir des itemsets fréquents 
rules = association_rules(itemsets_freq,metric="confidence",min_threshold=0.75)

#Affichage des 5 premières règles
Five_rules = rules.iloc[:5,:]
print("\n")
print("Les 5 premières règles sont : ")
print(Five_rules)

#Filtrage des règles en affichant celles qui présentent un LIFT supérieur ou égal à 7
print("\n")
print("Les règles qui présentent un LIFT supérieur ou égal à 7 sont : ")
print(rules[rules['lift'].ge(7.0)])

#Filtrage des règles menant au conséquent {‘2pct_milk’}
print("\n")
print("Les règles qui mènent au conséquent '2pct_milk' sont : ")
print(rules[rules['consequents'].eq({'2pct_Milk'})])
