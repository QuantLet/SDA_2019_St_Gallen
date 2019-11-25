# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:47:38 2019

@author: KAT
Input: X_train, y_train, X_test, y_test (Outlier_detection.py)
Output: 
Purpose:
"""

import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import matplotlib.pyplot as plt

score={'AUC':'roc_auc', 
           'RECALL':'recall',
           'PRECISION':'precision',
           'F1':'f1'}

X_train = pd.read_csv(r"C:\Users\sdauser\Documents\ML\drive-download-20191121T164109Z-001\X_all_train_wo_OS.csv", low_memory=False, index_col = 0).iloc[:, 2:]
y_train = pd.read_csv(r"C:\Users\sdauser\Documents\ML\drive-download-20191121T164109Z-001\y_train_wo_OS.csv", low_memory=False, index_col = 0).squeeze()
X_test = pd.read_csv(r"C:\Users\sdauser\Documents\ML\drive-download-20191121T164109Z-001\X_all_test.csv", low_memory=False, index_col = 0).iloc[:, 2:]
y_test = pd.read_csv(r"C:\Users\sdauser\Documents\ML\drive-download-20191121T164109Z-001\y_test.csv", low_memory=False, index_col = 0).squeeze()

LogReg = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', LogisticRegression(solver='lbfgs', random_state=0))
            ])
LogReg_para = {}
RandF = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', RandomForestClassifier(random_state=0))
            ])
RandF_para = {'classification__n_estimators':[20, 50, 100, 200, 400, 800], 'classification__max_depth':[2, 5, 10, 20]}
AdaBoost = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', AdaBoostClassifier(random_state=0))
            ])
AdaBoost_para = {'classification__n_estimators':[20, 50, 100, 200, 400, 800]}
SVM = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', SVC(decision_function_shape='ovr', degree=3, gamma='auto'))
            ]) 
SVM_para = {'classification__C':[0.01, 0.1, 1, 10], 'classification__kernel':('linear', 'rbf')}
NaivBay = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', GaussianNB())
            ])
NaivBay_para = {}
Knn = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski'))
            ])
Knn_para = {'classification__n_neighbors': (10, 15, 25)}

clasifier_names = ["Logistic Regression", "Random Forest", "Adaptive Boosting", "Support Vector Machines", "Naive Bayes", "K Nearest Neighbours"]
classifiers = [LogReg, RandF, AdaBoost, SVM, NaivBay, Knn]
parameters = [LogReg_para, RandF_para, AdaBoost_para, SVM_para, NaivBay_para, Knn_para]

results = list()

for i in range(len(classifiers)):
    clf = GridSearchCV(classifiers[i], parameters[i], cv=5, scoring=score, n_jobs=-1, refit=False, return_train_score=True)
    clf.fit(X_train, y_train)
    results.append([clasifier_names[i], clf.cv_results_])
    print(clasifier_names[i])
    print(clf.cv_results_)

with open("results_ML.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(results)
     
label, test_precision, test_recall, train_precision, train_recall = list(),list(),list(),list(),list()
for result in results:
    label.append(result[0])
    test_precision.extend([result[1]['mean_test_PRECISION'].tolist()])
    test_recall.extend([result[1]['mean_test_RECALL'].tolist()])
    train_precision.extend([result[1]['mean_train_PRECISION'].tolist()])
    train_recall.extend([result[1]['mean_train_RECALL'].tolist()])

colors = {"Logistic Regression":"red", "Random Forest":"blue", "Adaptive Boosting":"green", "Naive Bayes":"orange", "Support Vector Machines":"black", "K Nearest Neighbours":"purple"}

fig, ax = plt.subplots(figsize=(15,15))
for i in range(len(label)):
    ax.scatter(test_recall[i], test_precision[i], c=colors[label[i]], label=label[i])
ax.axis((0,1,0,1))
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
plt.legend()
plt.title("Test scores: Recall vs. Precision")
plt.savefig("test_scores_recall_vs_precision.png", transparent=True)
plt.show()

fig, ax = plt.subplots(figsize=(15,15))
for i in range(len(label)):
    ax.scatter(train_recall[i], train_precision[i], c=colors[label[i]], label=label[i])
ax.axis((0,1,0,1))
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
plt.legend()
plt.title("Train scores: Recall vs. Precision")
plt.savefig("train_scores_recall_vs_precision.png", transparent=True)
plt.show()