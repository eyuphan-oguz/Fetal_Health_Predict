import BayesianOptimization as BayesianOptimization
import sns as sns
from sklearn import tree
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt


def simulatedAnnealing(xTrain,yTrain,model,modelName,yTestData):

    n_features = xTrain.shape[1]  ## n_features değişkenin içine shape methodunu kullanarak sütün sayısını atar içine yazılan 1 değeri sütün sayısını verirken 0 değeri yazılırsa satır sayısını döndürür
    selected_features = []
    T = 1000
    cooling_rate = 2
    best_accuracy = 0
    while T > 1:
        feature = random.randint(0, n_features - 1)
        selected_features.append(feature)
        #print(selected_features)
        model.fit(xTrain[:, selected_features], yTrain)
        y_pred = model.predict(xTest[:, selected_features])
        accuracy = accuracy_score(yTestData, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            selected_features.remove(feature)
            p = np.exp((accuracy - best_accuracy) / T)
            r = random.uniform(0, 1)
            if p > r:
                best_accuracy = accuracy
        T -= cooling_rate
    # Print the list of selected features
    #print(selected_features)
    # no_duplicates_list = list(set(selected_features))
    # print(no_duplicates_list)
    cm = confusion_matrix(yTestData, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Optimizasyon Sonrası {}".format(modelName))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("Simulated Annealing uygulanmış {} için doğruluk oranı = {}".format(modelName,best_accuracy))




data = pd.read_csv("fetal_health.csv")

train, test = train_test_split(data, test_size=0.2 , random_state=42)

# Son sütun hariç tüm satırları ve sütunları seç
xTrain = train.iloc[:, :-1]
# Son sütunu seç
yTrain = train.iloc[:, -1]

# Son sütun hariç tüm satırları ve sütunları seç
xTest = test.iloc[:, :-1]
# Son sütunu seç
yTest = test.iloc[:, -1]

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

gNB = GaussianNB()
gNB.fit(xTrain, yTrain)
y_predNB = gNB.predict(xTest)
print("Gaussian Naive Bayes için doğruluk oranı = {}".format(accuracy_score(yTest, y_predNB)))
gNBcm = confusion_matrix(yTest, y_predNB)

sns.heatmap(gNBcm, annot=True,fmt="d")
plt.title("Optimizasyon Öncesi Gaussian Naive Bayes")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(xTrain, yTrain)
y_predkNN = kNN.predict(xTest)
print("K Nearest Neighbors için doğruluk oranı = {}".format(accuracy_score(yTest, y_predkNN)))
kNNcm = confusion_matrix(yTest, y_predkNN)

sns.heatmap(kNNcm, annot=True,fmt="d")
plt.title("Optimizasyon Öncesi K Nearest Neighbors")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


dT = tree.DecisionTreeClassifier()
dT = dT.fit(xTrain, yTrain)
y_preddT = dT.predict(xTest)
print("decision tree için doğruluk oranı = {}".format(accuracy_score(yTest, y_preddT)))
dTcm = confusion_matrix(yTest, y_preddT)

sns.heatmap(dTcm, annot=True,fmt="d")
plt.title("Optimizasyon Öncesi Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


model = GaussianNB()
simulatedAnnealing(xTrain,yTrain,model,"Gaussian Naive Bayes",yTest)

modelKNN = KNeighborsClassifier()
simulatedAnnealing(xTrain,yTrain,modelKNN,"KNN",yTest)

modelDT = tree.DecisionTreeClassifier()
simulatedAnnealing(xTrain,yTrain,modelDT,"Decision Tree",yTest)






