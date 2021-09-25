#!/usr/bin/env python
# coding: utf-8

# <h1>CREDIT SCORING PAR LA PRATIQUE (Partie 1)</h1>
# 
# **Yves KOUAKOU** Consultant et formateur en Data Science                  

# 
# **1. [Introduction](#Introduction)** <br>
# 
# **2. [Exploration générale](#Known)** <br>
# 
# **3. [Prétraitement 1](#pretraitement)** <br>
# 

# <a id="Introduction"></a> <br>
# # **1. Introduction** 
# 
# L'objectif de ce projet est de développer un modèle de crédit scoring qui pourra être utilisé pour l’octroi de crédits dans une banque ou un établissement spécialisé de crédit.
# 
# Nous allons utiliser un jeu de données public pour construire cet outil de crédit scoring.
# Le jeu de données est disponible sur internet <b>kaggle.com </b> et sur mon compte GitHub (…..).  Il contient 1000 dossiers de crédits à la consommation, dont 300 impayés. 
# 
# Dans cet ensemble de données, chaque entrée représente une personne qui prend un crédit auprès d'une banque. Chaque personne est classée comme un bon ou un mauvais risque de crédit selon l'ensemble des attributs. 

# <a id="donnees"></a> <br>
# #  Données
# 

# In[3]:


# Les librairies 
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 


# In[4]:


donnee = pd.read_csv("german_credit_data.csv",index_col=0) # Utilisation de La librairie pandas pour importer notre donnée
donnee # Vérifier si notre jeux données a été bien importé


# <h2>Descriptions des Variables</h2>
# <b>Age </b>(numeric)<br>
# <b>Sex </b>(male, female)<br>
# <b>Job </b>(numeric): 
#     - 0 (non qualifié and non resident), 
#     - 1 (non qualifié and resident), 
#     - 2 (qualifié)
#     - 3 (très qualifié))
# <b>Housing</b> (own, rent, or free)<br>
# <b>Saving accounts (compte épargne)</b> (little, moderate, quite rich, rich)<br> 
# <b>Checking account (compte courant)</b> (numeric)<br>
# <b>Credit amount (crédit emprunte)</b> (numeric)<br>
# <b>Duration</b> (numeric, in month)<br>
# <b>Purpose</b>(text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others<br>
# <b>Risk </b> (Variable cible - Good ou Bad Risk)<br>

# Nous constatons que la donnée a été bien chargée

# <a id="Known"></a> <br>
# # **2. Exploration générale de notre donnée:** 
# 
# - Variables
# - Données manquantes
# 

# In[3]:


donnee.info()


# - On constate qu'il y a 1000 clients et 10 variables. Les variables **Saving accounts** et **Checking account** ont des données manquantes.
# - Il y a 4 variables de types integer : **Age, Job, Credit amount, Duration**
# - 6 variables de types object : **Sex, Housing, Saving accounts, Checking account,Purpose, Risk(variable cible)**
# 

# # Données Manquantes

# In[5]:


# fonction permetant de compter le nombre de variables manquantes et de calculer le pourcentange des 
# donnees manquantes de chaque variables

def missing_value_table(df):
    missing_value = df.isna().sum().sort_values(ascending=False)
    missing_value_percent = 100 * df.isna().sum()//len(df)
    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Données manquantes', 1 : '% Taux'})
    cm = sns.light_palette("purple", as_cmap=True)
    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)
    return missing_value_table_return
  
missing_value_table(donnee)


# 
# * **Checking account** : il y a 394 valeurs manquantes soit 39% des données manquantes pour cette variable
# * **Saving accounts** : il y a 183 valeurs manquantes soit 18% des données manquantes pour cette variable

# # Répartition par genre et selon les classes de la variable cible 

# In[6]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.countplot(donnee['Sex'], ax=ax[0]).set_title('Répartition par genre : Male - Female');
sns.countplot(donnee.Risk, ax=ax[1]).set_title('Répartition variable cible Risk : Good - Bad');


# * Le nombre d'homme est pratiquement le double de celui des femmes
# * Il y a plus de clients de bon clients que de mauvais.

# ## Représentation des 3 variables continues non découpées en classes 
# 

# - Segmentation de nos clients en **good** et **bad** clients selon la variable cible **Risk**

# In[7]:


df_good = donnee[donnee["Risk"] == 'good']
df_bad = donnee[donnee["Risk"] == 'bad']


# ## Nous allons utiliser seaborn pour la visualisation des distributions
# Pour la documentation voir : www.seaborn.com

# # Age

# In[8]:


fig, ax = plt.subplots(nrows=2, figsize=(12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

# représentation des distributions utilisant les couleurs green et red selon le segment
g1=sns.distplot(df_good["Age"], ax=ax[0], color="g",)
g2=sns.distplot(df_bad["Age"], ax=ax[1], color="r")

# Titre des distributions
g1.set_title("Distribution Age : Pas d'impayé", fontsize=15)
g2.set_title("Distribution Age : Présence d'impayé", fontsize=15)

print("Distribution de l'âge du demandeur (sur les bons clients et les autres) ")

print(" La Moyenne d'âge des clients ayant des Impayés :",df_bad["Age"].mean())  
print(" La Moyenne d'âge des clients n'ayant pas des Impayés :",df_good["Age"].mean())  


# # Credit amount

# In[9]:


fig, ax = plt.subplots(nrows=2, figsize=(12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

#
g3=sns.distplot(df_good["Credit amount"], ax=ax[0], color="g",kde=True)
g4=sns.distplot(df_bad["Credit amount"], ax=ax[1], color="r")

# Titre des distributions
g3.set_title("Distribution Credit amount : Pas d'impayé", fontsize=15)
g4.set_title("Distribution Credit amount : Présence d'impayé", fontsize=15)

print("Distribution du montant du crédit (sur les bons clients et les autres) ")

print(" La Moyenne du montant de crédit chez les clients ayant des Impayés :",df_bad["Credit amount"].mean())  
print(" La Moyenne du montant de crédit chez les clients n'ayant pas des Impayés :",df_good["Credit amount"].mean())  


# # Duration

# In[10]:


fig, ax = plt.subplots(nrows=2, figsize=(12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)


g5=sns.distplot(df_good["Duration"], ax=ax[0], color="g",bins=15)
g6=sns.distplot(df_bad["Duration"], ax=ax[1], color="r",bins=15)

g5.set_title("Distribution Duration  : Pas d'impayé", fontsize=15)
g6.set_title("Distribution Duration  : Présence d'impayé", fontsize=15)
print("Distribution de la durée (en mois) de crédit (sur les bons clients et les autres) ")


print(" La moyenne de la durée du crédit chez les clients ayant des Impayés :",df_bad["Duration"].mean())  
print(" La moyenne de la durée du crédit chez les clients pas des Impayés :",df_good["Duration"].mean())  


# # Interprétation

# - L'âge des clients est compris entre 19 et 75 ans, la distribution des crédits sans impayés est plus homogènes que pour les autres. Les clients ayant moins de 40 ans sont ceux qui ont le plus d'impayés
# - Les crédits de très petit montant sont rares, le minimum est de 250 euros et 95% dépassent 700 euros. On atteint une fréquence maximale autour de 1250 euros qui ne fait que baisser par la suite. 
# - Ceux qui ont des impayés ont la plus forte proportion de montant plus élevé.
# - La durée du crédit présente des pics à 12, 25, 35, 48 et 60 mois. La plus forte proportion de crédits plus longs est parmi ceux qui ont des impayés.
# - On constat que les 3 variables ont une liaison avec la variable cible **Risk** donc nous allons les discrétiser

# <a id="pretraitement"></a> <br>
# # **3. Prétraitement** 

# <a id="definition"></a> <br>
# # **3.1 Qu'est ce que la discrétisation ?**
# 

# - La discrétisation consiste à transformer une variable quantitative en une variable qualitative ordinale. 
# - Pour discrétiser une variable quantitative, il suffit de la découper en classes (intervalles).

# <a id="why"></a> <br>
# # **3.2 Pourquoi discrétiser une variable ?**

# Nous allons discrétiser les variables non regroupées en classes, car certaines analyses ne fonctionnent qu'avec les variables qualitatives regroupées en classes. Par exemple :
# - L'analyse des correspondances multiples (**ACM**) qui est une méthode d'analyse factorielle adaptée aux données qualitatives (aussi appelées catégorielles)

# - **Nous utiliserons cette technique plus bas sur les clients (individus) et sur les variables** 
# - L'ACM nous permettra d'étudier les ressemblances entre individus du point de vue de l'ensemble des variables 
# et de dégager les profils d'individus. Elle permettra également de faire un bilan des liaisons entre variables et d'étudier les associations de modalités.

# <a id="discretisation"></a> <br>
# # **3.3 Discrétisation**

# * Vérifions si toutes les variables ont été discrétisées (découpées en classe)

# In[35]:


columns = ["Age","Sex","Job","Housing","Saving accounts","Checking account","Credit amount",
           "Duration","Purpose","Risk"]

def unique_value(donnee, column_name):
    return donnee[column_name].nunique()

print("Nombre de valeur unique :\n",unique_value(donnee, columns))


# 

# - Seules 3 variables (**Age, credit account, Duration**) n'ont pas été découpées en classe, nous allons commencer par les discrétiser.
# - La discrétisation des variables permettra de les utiliser conjointement aux autres à l'aide des mêmes méthodes. Ce qui procurera plus de simplicité et de lisibilité.

# <a id="methode"></a> <br>
# # **3.3.1 Méthode**

# ## Discrétisation des variables par la méthode des déciles

# In[11]:


df=donnee.copy() # une copie de la données originale


# In[13]:


# Les Déciles
deci=[]
for i in range(0,10):
    deci.append(i)
deci


# Nous bien 10 classes numérotée de 0 à 9

# ## Fonction pour compter le nombre d'observation par décile 
#       (Découper la variable en 10 tranches d'effectifs égaux)

# In[14]:


# Création d'une nouvelle variable appelée décile
df['Decile'] = pd.qcut(df['Age'], 10, labels=False)

# Nombre de clients par décile
nbreObs=[]
for i in range(0,10):
    s=df[df["Decile"]== i]
    a=s.shape[0]
    nbreObs.append(a)

# Tableau
TabNbreDecile=[deci,nbreObs]
np.array(TabNbreDecile).T


# - Il y a plus de 100 clients dans les déciles 0, 1, 4, 5, et 7.
# - Le décile 1 contient le plus de clients.

# <a id="variables"></a> <br>
# # **3.3.2 Discrétisation des variables**

# Nous allons disrétiser les variables numériques continues : **Age, credit account et Duration**

# # AGE

# In[15]:


# découper l'âge en décile dans une nouvelle variable "DecileA"
df['DecileA'] = pd.qcut(df['Age'], 10, labels=False) 


# In[16]:


# Tableau indiquant la classe d'appartenance de chaque modalité de la variable âge 
df[['Age','DecileA']] 


# * Fonction permettant d'identifier le minimum dans chaque décile

# In[17]:


mini=[]
for i in range(0,10):
    s=df[df["DecileA"]== i]
    a=s["Age"].min()
    mini.append(a)


# * Fonction permettant d'identifier le maximum dans chaque décile

# In[18]:


maxi=[]
for i in range(0,10):
    s=df[df["DecileA"]== i]
    a=s["Age"].max()
    maxi.append(a)


# * Tableau des seuils des déciles 

# In[19]:


#Tableau
F=[deci,nbreObs,mini,maxi] 
tab=np.array(F)
tab.T


# - La colonne 1 représente les déciles
# - La colonne 2 représente le nombre d'observations par décile
# - La colonne 3 représente l'âge minimum par décile
# - La colonne 4 représente l'âge maximal par décile

# ## Comment lire le tableau ci-dessus
# Exemple : La deuxième ligne représente le décile 2 et on peut dire : **Le décile 1 (ou la classe 2) contient 135 clients dont l'âge vari entre 24 et 26 ans etc....**

# ## Tableau de contingence des déciles de l'âge et la variable cible Risk

# In[20]:


cm = sns.light_palette("lightgreen", as_cmap=True)
tab=pd.crosstab(df['Decile'],df.Risk).style.background_gradient(cmap = cm)
tab


# ## Taux d'impayé pour chaque décile

# In[21]:


taux_bad=[]
for i in range(0,10):
    s=df[df["Decile"]== i]
    bad=pd.DataFrame(s["Risk"]=="bad")
    pba=100 * bad.sum()//len(s["Risk"])
    taux_bad.append(pba)


# In[22]:


dec=np.array(deci).reshape(10,1) # changer les dimensions pour faciliter la suite


# * Taux d'impayé pour chaque décile

# In[23]:


print(np.concatenate((dec, np.array(taux_bad)), axis=1))


# # Décision
# - Le tableau montre clairement que les deux premiers déciles de l'âge ont un taux d'impayés supérieur aux taux  d'impayés des autres déciles. L'âge maximum du second décile étant 26 ans, il y a donc un seuil à 26 ans. 
# - Le taux d'impayé des autres déciles varie entre 20% et 30%. Donc on décide découper l'âge en deux classes. Ainsi nous venons de déterminer le nombre de classes optimales pour la variables âge. 

# # Credit amount ( montant du prêt )
# 
#      Ici nous utiliserons les vingtiles (découper la variable en 20 tranches d'effectifs égaux)

# In[24]:


#pd.set_option("max_rows", 10) # affiche toutes les lignes
df['vingtiles'] = pd.qcut(df['Credit amount'], 20, labels=False) # découper le montant du prêt en 20 parties
df[['Credit amount','vingtiles']]


# * Fonction permettant d'identifier le minimum dans chaque vingtile

# In[25]:


# Vingtiles
vingtile=[]
for i in range(0,20):
    vingtile.append(i)


# In[26]:


#  Nombre de clients par vingtile
nbreObsCE=[]
for i in range(0,20):
    s=df[df["vingtiles"]== i]
    a=s.shape[0]
    nbreObsCE.append(a)


# In[27]:


# Fonction permettant d'identifier le minimum dans chaque vingtile
mini=[]
for i in range(0,20):
    s=df[df["vingtiles"]== i]
    a=s["Credit amount"].min()
    mini.append(a)


# In[28]:


# Fonction permettant d'identifier le maximum dans chaque vingtile
maxi=[]
for i in range(0,20):
    s=df[df["vingtiles"]== i]
    a=s["Credit amount"].max()
    maxi.append(a)


# In[29]:


# Tableau
F=[vingtile,nbreObsCE,mini,maxi] 
tab=np.array(F)
print(tab.T)


# - La colonne 1 représente les Vingtiles
# - La colonne 2 représente le nombre d'observations par vingtile
# - La colonne 3 représente le montant minimum par vingtile
# - La colonne 4 représente le montant maximal par vingtile

# ## Comment lire le tableau ci-dessus
# Exemple : La deuxième ligne représente le vingtile 1 et on peut dire : **Le vingtile 1 (ou la classe 2) contient 51 clients dont le montant du crédit varie entre 709 et 932 euros etc....**

# ## Tableau de contingence des vingtiles de montant et la variable cible Risk

# In[30]:


cm = sns.light_palette("lightgreen", as_cmap=True)
tab=pd.crosstab(df['vingtiles'],df.Risk).style.background_gradient(cmap = cm)
tab


# ## Taux impayé pour chaque vingtile
# 

# In[31]:


taux_badV=[]
for i in range(0,20):
    s=df[df["vingtiles"]== i]
    bad=pd.DataFrame(s["Risk"]=="bad")
    pba=100 * bad.sum()//len(s["Risk"])
    taux_badV.append(pba)


# In[32]:


ving=np.array(vingtile).reshape(20,1) # changer les dimensions pour faciliter la suite


# In[33]:


np.concatenate((ving, np.array(taux_badV)), axis=1)


# # Décision 
# 
# - Après avoir découpé la variable en 20 tranches d'effectifs égaux puis mesuré les taux d'impayés dans chacune des tranches. Nous constatons qu'il y a un seuil net entre le 14ème et le 15ème vingtile qui correspond à 3972 euros que nous allons arrondir à 4000 euros. 
# - Donc on décide découper le montant du crédit en deux classes. Ainsi nous venons de déterminer le nombre de classes optimales pour la variables credit amount. 

# 

# # Duration (Durée du crédit)
# 
#      Ici nous utiliserons les déciles (découper la variable en 2 tranches d'effectifs égaux)
# 
# 

# In[34]:


# Tableau indiquant la classe d'appartenance de chaque modalité de la variable duration 

df['DecileD'] = pd.qcut(df['Duration'], 10, labels=False,duplicates='drop') 
df[['Duration','DecileD']]


# In[35]:


# Déciles
DecileD=[]
for i in range(0,10):
    DecileD.append(i)


# In[36]:


#  Nombre de durées par décile
nbreObsDuration=[]
for i in range(0,10):
    s=df[df["DecileD"]== i]
    a=s.shape[0]
    nbreObsDuration.append(a)


# In[37]:



# Fonction permettant d'identifier le minimum dans chaque décile
mini=[]
for i in range(0,10):
    s=df[df["DecileD"]== i]
    a=s["Duration"].min()
    mini.append(a)


# Fonction permettant d'identifier le maximum dans chaque vingtile
maxi=[]
for i in range(0,10):
    s=df[df["DecileD"]== i]
    a=s["Duration"].max()
    maxi.append(a)


# In[39]:


# Tableau 
F=[DecileD,nbreObsDuration,mini,maxi] 
tab=np.array(F)
print(tab.T)


# - La colonne 1 représente les déciles
# - La colonne 2 représente le nombre d'observations par décile
# - La colonne 3 représente le montant minimum par décile
# - La colonne 4 représente le montant maximal par décile
# ## Comment lire le tableau ci-dessus
# Exemple : La deuxième ligne représente le décile 1 et on peut dire : **Le décile 1 (ou la classe 2) contient 216 clients dont la durée du crédit varie entre 10 et 12 mois etc....**
# 
# La durée maximale étant 72 mois, il est normal d'avoir des **nan** (donnée manquantes dans ce tableau) apres cette date.

# In[40]:


## Tableau de contingence des déciles de la durée et la variable cible Risk

cm = sns.light_palette("lightgreen", as_cmap=True)
tab=pd.crosstab(df['DecileD'],df.Risk).style.background_gradient(cmap = cm)
tab


# In[41]:


## Taux impayé pour chaque décile

taux_badV=[]
for i in range(0,10):
    s=df[df["DecileD"]== i]
    bad=pd.DataFrame(s["Risk"]=="bad")
    pba=100 * bad.sum()//len(s["Risk"])
    taux_badV.append(pba)


# In[42]:


dim=np.array(DecileD).reshape(10,1) # changer les dimensions pour faciliter la suite

print('Taux impayé pour chaque décile \n',np.concatenate((dim, np.array(taux_badV)), axis=1))


# # Décision 
# 

# - Le tableau montre clairement que les trois premiers déciles de la durée ont un taux d'impayés compris entre 18 et 25 %. Et les déciles 3 à 5 ont des taux compris entre 25 et 33 %. Les deux derniers déciles ont des taux d'impayés au dessus des 35%. Donc on décide de découper la variable duration en trois classes. Ainsi nous venons de déterminer le nombre de classes optimales pour cette variable (de 0 à 15 mois, de 16 à 36 mois et ceux supérieur à 36 mois). 

# # Applications : Discrétisation 

# In[43]:


# AGE
Cat_Age = []
for i in df["Age"]:
    if i<=25:
        Cat_Age.append("0-25 ans")
    elif i>25:
        Cat_Age.append("26-75 ans")
        
df["Cat_Age"] = Cat_Age 


# In[44]:


# Credit amount
Cat_Credit_Amount = []
for i in df["Credit amount"]:
    if i<=4000:
        Cat_Credit_Amount.append("<= 4000 Euros")
    elif i>4000:
        Cat_Credit_Amount.append(" > 4000 Euros")
        
df["Cat_Credit_Amount"] = Cat_Credit_Amount 


# In[45]:


# Duration
Cat_Duration = []
for i in df["Duration"]:
    if (i<16):
        Cat_Duration.append("0 - 15 mois")
    elif (i>=16) and (i<37):
        Cat_Duration.append("16-36 mois")  
    elif (i>36) :
        Cat_Duration.append("> 36 mois")  
df["Cat_Duration"] = Cat_Duration


# In[46]:


# Aperçu de notre jeu de données après les modifications apportées 
df


# ## Nous constatons qu'il y a 17 variables au lieu de 10 donc supprimons les variables non discrétiser

# In[47]:


df_Discretisé=df.copy() #une copy pour garder une trace 


# In[49]:


df_Discretisé.drop(['Duration','Age','Credit amount',"Decile","DecileA","vingtiles","DecileD"], axis = 1, inplace=True)
print("Donnée à utiliser pour la suite du projet")
df_Discretisé


# ## Toutes les variables étant disponibles sous forme de classes, nous pouvons effectuer une analyse des correspondant multiples

# - Merci, et à bientôt pour la suite

# In[ ]:




