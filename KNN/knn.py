import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("cars.data")
#print(data.head())

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform((list(data['lug_boot'])))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

x = list(zip(buying, maint, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#print(x_train[0], y_train[0])

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("Accuracy :", acc)

predictions = model.predict(x_test)
class_names = ['unacc', 'acc', 'good', 'vgood']

for i in range(len(predictions)):
    print("Data:", x_test[i], "| Actual Class:", class_names[y_test[i]], "| Prediction:", class_names[predictions[i]])

    n = model.kneighbors([x_test[i]], 9, True)
    print("N :", n)
