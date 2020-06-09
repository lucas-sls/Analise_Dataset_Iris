from Data_Exploration_1 import sns, df, plt

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
