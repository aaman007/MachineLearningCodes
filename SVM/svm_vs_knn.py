import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train, y_train)
classes = ['malignant', 'benign']

svm_model = svm.SVC(kernel="linear", C=2)
svm_model.fit(x_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=9)
knn_model.fit(x_train, y_train)

svm_y_pred = svm_model.predict(x_test)
knn_y_pred = knn_model.predict(x_test)

svm_acc = metrics.accuracy_score(svm_y_pred, y_test)
knn_acc = metrics.accuracy_score(knn_y_pred, y_test)

print("SVM Accuracy :", svm_acc)
print("KNN Accuracy :", knn_acc)