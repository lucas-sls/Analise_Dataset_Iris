from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from Models_Analysis_3 import SVC, x_train, y_train, x_validation, y_validation

  # Fazendo as predições com o modelo escolhido:
svm = SVC(gamma='auto')
svm.fit(x_train, y_train)
predict = svm.predict(x_validation)

print(accuracy_score(y_validation, predict))
print(confusion_matrix(y_validation, predict))
print(classification_report(y_validation, predict))