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
print(df.shape) # 150 linhas e 5 colunas

    # Mostra primeiras 15 linhas
print(df.head(15))

    # Resumo dos atributos
print(df.describe())

    # Distribuição por classe
print(df.groupby('class').size()) # 50, 50 e 50


# VISUALIZAÇÃO DOS DADOS

    #Scatter-Plot + KDE
sns.set_style('darkgrid')
sns.pairplot(df, hue='class', height=2.5, palette='Set2', diag_kind='kde' )
plt.show()

     #Histograma dos atributos quantitativos
sns.set_style('darkgrid')
df.hist(bins=7)
plt.show()

    #Countplot das espécies de íris
sns.set_style('darkgrid')
sns.countplot(x='class',data=df, palette='Set2')
plt.title('Quantidade de amostras por Espécie de Iris', fontsize=16)
plt.xlabel('Espécies de Iris', fontsize=13)
plt.ylabel('Quantidade',fontsize=13)
plt.show()

    #Boxplots das variáveis quantitativas
sns.set_style('darkgrid')
fig = plt.figure()
fig.subplots_adjust(hspace=0.5, wspace=0.3)
ax = fig.add_subplot(2,2,1)
plt.title('Comp. de Sépala VS Tipo de Iris')
sns.boxplot(x='class',y='sepal-length', data=df, ax=ax)
ax = fig.add_subplot(2,2,2)
plt.title('Comp. de Pétala VS Tipo de Iris')
sns.boxplot(x='class',y='petal-length', data=df, ax=ax)
ax = fig.add_subplot(2,2,3)
plt.title('Larg. de Sépala VS Tipo de Iris')
sns.boxplot(x='class',y='sepal-width', data=df, ax=ax)
ax = fig.add_subplot(2,2,4)
plt.title('Larg. de Pétala VS Tipo de Iris')
sns.boxplot(x='class',y='petal-width', data=df, ax=ax)

plt.show()





