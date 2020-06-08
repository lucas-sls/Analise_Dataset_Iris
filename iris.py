# Carregando libraries

import pandas
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

    # Correlações entre as variáveis
print('\nESPÉCIE IRIS: \n',df.corr(),'\n')
df_setosa = df[df['class'] == 'Iris-setosa']
print('SETOSA: \n',df_setosa.corr(),'\n')
df_virginica = df[df['class'] == 'Iris-virginica']
print('VIRGINICA', df_virginica.corr(),'\n')
df_versicolor = df[df['class'] == 'Iris-versicolor']
print('VERSICOLOR: \n', df_versicolor.corr(), '\n')

     # Heatmap das correlações
sns.set_style('darkgrid')
fig = plt.figure(figsize=(10,6))
plt.suptitle('Análise de correlação entre atributos de uma mesma Iris',
             fontsize=17)
ax = fig.add_subplot(131)
plt.title('Iris Setosa')
sns.heatmap(df_setosa.corr(),annot=True, cmap='plasma',
            linecolor='gray', linewidths=1, ax=ax)
ax = fig.add_subplot(132)
plt.title('Iris Virginica')
sns.heatmap(df_virginica.corr(),annot=True, cmap='plasma',
            linecolor='gray', linewidths=1, ax=ax)
ax = fig.add_subplot(133)
plt.title('Iris Versicolor')
sns.heatmap(df_versicolor.corr(), annot=True, cmap='plasma',
            linecolor='gray', linewidths=1)
plt.subplots_adjust(top = 0.8, bottom=0.2, wspace=0.12)
plt.show()

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
ax_1 = fig.add_subplot(2,2,1)
plt.title('Comp. de Sépala VS Tipo de Iris')
sns.boxplot(x='class',y='sepal-length', data=df, ax=ax_1)
ax_1 = fig.add_subplot(2,2,2)
plt.title('Comp. de Pétala VS Tipo de Iris')
sns.boxplot(x='class',y='petal-length', data=df, ax=ax_1)
ax_1 = fig.add_subplot(2,2,3)
plt.title('Larg. de Sépala VS Tipo de Iris')
sns.boxplot(x='class',y='sepal-width', data=df, ax=ax_1)
ax_1 = fig.add_subplot(2,2,4)
plt.title('Larg. de Pétala VS Tipo de Iris')
sns.boxplot(x='class',y='petal-width', data=df, ax=ax_1)

plt.show()

    #ANÁLISE DOS ALGORITMOS

    #Separação do dataset - 80% para treinamento e 20% para testes!
arr = df.values

x = arr[:,0:4]
y = arr[:,4]
val_size = 0.20
seed = 7

x_train, x_validation, y_train, y_validation =\
    model_selection.train_test_split(x,y, test_size=val_size, random_state=seed)

    # Cross-Validation
seed = 7
score = 'accuracy'

modelos = []
modelos.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
modelos.append(('LDA', LinearDiscriminantAnalysis()))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('CART', DecisionTreeClassifier()))
modelos.append(('NB', GaussianNB()))
modelos.append(('SVM', SVC(gamma='auto')))

results = []
names = []

for name, model in modelos:
    k_fold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(model,
                                                 x_train,
                                                 y_train,
                                                 cv=k_fold,
                                                 scoring=score)
    results.append(cv_results)
    names.append(name)
    msgm = "%s: %f(%f)" %(name, cv_results.mean(), cv_results.std())
    print(msgm)


    #Comparando eficiência dos algoritmos
sns.set_style('darkgrid')
fig = plt.figure()
plt.suptitle('Comparação de eficiência dos algoritmos aplicados')
ax = fig.add_subplot(111)
sns.boxplot(x=names,y=results, ax=ax)
plt.show()

    #Fazendo as predições com o modelo escolhido:
svm = SVC(gamma='auto')
svm.fit(x_train, y_train)
predict = svm.predict(x_validation)

print(accuracy_score(y_validation, predict))
print(confusion_matrix(y_validation, predict))
print(classification_report(y_validation, predict))

