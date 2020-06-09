    # Carregando libraries
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

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






