from Data_Exploration_1 import df, sns, plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#ANÁLISE DOS ALGORITMOS

    #Separação do dataset - 80% para treinamento e 20% para testes!
arr = df.values

x = arr[:, 0:4]
y = arr[:, 4]
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
sns.boxplot(x=names, y=results, ax=ax)
plt.show()
