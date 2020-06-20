import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = 'G3'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best_acc = 0

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# for _ in range(40):
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     model = linear_model.LinearRegression()
#     model.fit(X_train, y_train)
#
#     accuracy = model.score(X_test, y_test)
#     print("Accuracy :", accuracy)
#
#     if accuracy > best_acc:
#         best_acc = accuracy
#         with open("student_model.pickle", "wb") as file:
#             pickle.dump(model, file)

pickle_in = open("student_model.pickle", "rb")
model = pickle.load(pickle_in)
print(model.score(X_test, y_test))

print("Co : ", model.coef_)
print("Intercepts :", model.intercept_)

predictions = model.predict(X_test)

for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])


style.use("ggplot")
p = "G2"
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()