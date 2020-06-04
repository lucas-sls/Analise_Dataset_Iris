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

#Gráficos de uma variável

sns.set_style('darkgrid')
sns.boxplot(x='class',y='sepal-length', data=df)
plt.title("Boxplot: Comp. de Sépala VS Espécies de Iris", fontsize=16)
plt.xlabel("Espécies de Iris", fontsize=12)
plt.ylabel("Comprimento de Sépala (cm)" , fontsize=12)
plt.show()

