# Carregando libraries

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


    # CARGA DE DADOS

# Link do arquivo csv
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

# Definindo o nome da coluna dos atributos
names = ['sepal-length', 'sepal-width', 'petal-length','petal-width','class']

# Carregando dataset com a url armazenada e com as colunas nomeadas
df = pandas.read_csv(url, names=names)

# Criando Dataframes por classe de íris, para análises futuras
df_setosa = df[df['class'] == 'Iris-setosa']
df_versicolor = df[df['class'] == 'Iris-virginica']
df_virginica = df[df['class'] == 'Iris-versicolor']


# ANÁLISE EXPLORATÓRIA

    # Número de linhas e colunas
#print(df.shape) # 150 linhas e 5 colunas

    # Mostra primeiras 15 linhas
#print(df.head(15))

    # Resumo dos atributos
#print(df.describe())

    # Distribuição por classe
#print(df.groupby('class').size()) # 50, 50 e 50


# VISUALIZAÇÃO DOS DADOS

    #Boxplots

# sns.set_style('darkgrid')
# sns.boxplot(x='class',y='sepal-length', data=df)
# plt.title("Boxplot: Comp. da Sépala VS Espécies de Iris", fontsize=16)
# plt.xlabel("Espécies de Iris", fontsize=12)
# plt.ylabel("Comprimento da Sépala (cm)" , fontsize=12)
# plt.show()

# sns.set_style('darkgrid')
# sns.boxplot(x='class',y='sepal-width', data=df)
# plt.title("Boxplot: Larg. da Sépala VS Espécies de Iris", fontsize=16)
# plt.xlabel("Espécies de Iris", fontsize=12)
# plt.ylabel("Largura da Sépala (cm)" , fontsize=12)
# plt.show()

# sns.set_style('darkgrid')
# sns.boxplot(x='class',y='petal-length', data=df)
# plt.title("Boxplot: Comp. da Pétala VS Espécies de Iris", fontsize=16)
# plt.xlabel("Espécies de Iris", fontsize=12)
# plt.ylabel("Comprimento da Pétala (cm)" , fontsize=12)
# plt.show()

# sns.set_style('darkgrid')
# sns.boxplot(x='class',y='sepal-width', data=df)
# plt.title("Boxplot: Larg. da Pétala VS Espécies de Iris", fontsize=16)
# plt.xlabel("Espécies de Iris", fontsize=12)
# plt.ylabel("Largura da Sépala (cm)" , fontsize=12)
# plt.show()


    #Histogramas
sns.set_style('darkgrid')
sns.distplot(a=df_setosa['sepal-length'])
plt.show()

sns.distplot(a=df_setosa['petal-length'])
plt.show()

sns.distplot(a=df_setosa['sepal-width'])
plt.show()

sns.distplot(a=df_setosa['petal-width'])
plt.show()

# sns.set_style('whitegrid')
# sns.distplot(a=df['sepal-width'])
# plt.show()

# sns.set_style('whitegrid')
# sns.distplot(a=df['petal-length'])
# plt.show()
#
# sns.set_style('whitegrid')
# sns.distplot(a=df['petal-width'])
# plt.show()

