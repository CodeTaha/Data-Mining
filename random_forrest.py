# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:08:22 2023
https://miracozturk.com/python-ile-siniflandirma-analizleri-rastgele-orman-random-forest-algoritmasi/
@author: Izoly V90
"""

from sklearn import datasets
import pandas as pan
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plot
import seaborn as sea

iris_dataset = datasets.load_iris()
print(iris_dataset)
print(iris_dataset.target_names)
print(iris_dataset.feature_names)

print(iris_dataset.data[0:10])
print(iris_dataset.target)

salt_data=pan.DataFrame({
       'sepal length':iris_dataset.data[:,0],
       'sepal width':iris_dataset.data[:,1],
       'petal length':iris_dataset.data[:,2],
       'petal width':iris_dataset.data[:,3],
       'species':iris_dataset.target
})

salt_data.head()

X=salt_data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y=salt_data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True)

clf=RandomForestClassifier(n_estimators=120)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Accuracy Value:",metrics.accuracy_score(y_test, y_pred))

# Örnek tahmin
# clf.predict([[1, 2, 3, 4]])

#Özelliklere yönelik önem puanlarını değerlendirelim. Öznitelik çıkarımı.
feature_imp = pan.Series(clf.feature_importances_,index=iris_dataset.feature_names).sort_values(ascending=False)
print(feature_imp)
sea.barplot(x=feature_imp, y=feature_imp.index)
plot.xlabel('Feature Importance Score')
plot.ylabel('Features')
plot.title('Visualizing Important Features')
plot.show()

X=salt_data[['petal length', 'petal width','sepal length']]
y=salt_data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75,shuffle=True, random_state=5)

clf=RandomForestClassifier(n_estimators=120)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Accuracy Value:",metrics.accuracy_score(y_test, y_pred))
