import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

def train_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    return accuracy, f1, report

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

num_training_samples = train_df.shape[0]  
print("Number of rows in training: ", num_training_samples)
num_test_samples = test_df.shape[0]       
print("Number of rows in test: ", num_test_samples)
num_features = train_df.shape[1] - 2    
print("Number of features: ", num_features)

features = train_df.columns[:-1]
print(features)

X_train = train_df.drop(['subject', 'Activity'], axis=1)
y_train = train_df['Activity']
X_test = test_df.drop(['subject', 'Activity'], axis=1)
y_test = test_df['Activity']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=280) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

rfc = RandomForestClassifier().fit(X_train_scaled, y_train)
model_selector = SelectFromModel(rfc, prefit=True, threshold='mean')
X_train_reduced = model_selector.fit_transform(X_train_scaled)
X_test_reduced = model_selector.transform(X_test_scaled)

feature_importances = rfc.feature_importances_

sorted_idx = np.argsort(feature_importances)
print(sorted_idx)
print(len(sorted_idx))
plt.figure(figsize=(10, 8))
plt.barh(np.array(features)[sorted_idx[0:50]], feature_importances[sorted_idx[0:50]], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.show()

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
accuracy_scores = {}
f1_scores = {}
classification_reports = {}

for kernel in kernels:
    svm_model = SVC(kernel=kernel, C=1, gamma='auto')
    
    accuracy, f1, report = train_evaluate(svm_model, X_train_scaled, X_test_scaled, y_train, y_test)
    accuracy_scores[f'SVM_{kernel}'] = accuracy
    f1_scores[f'SVM_{kernel}'] = f1
    classification_reports[f'SVM_{kernel}'] = report
    print(f"Kernel: {kernel}, Accuracy: {accuracy}")

    accuracy, f1, report = train_evaluate(svm_model, X_train_reduced, X_test_reduced, y_train, y_test)
    accuracy_scores[f'SVM_{kernel}_Reduced'] = accuracy
    f1_scores[f'SVM_{kernel}_Reduced'] = f1
    classification_reports[f'SVM_{kernel}_Reduced'] = report
    print(f"Kernel: {kernel}, Reduced Accuracy: {accuracy}")
    
    accuracy, f1, report = train_evaluate(svm_model, X_train_pca, X_test_pca, y_train, y_test)
    accuracy_scores[f'SVM_{kernel}_PCA'] = accuracy
    f1_scores[f'SVM_{kernel}_PCA'] = f1
    classification_reports[f'SVM_{kernel}_PCA'] = report
    print(f"Kernel: {kernel}, PCA Accuracy: {accuracy}")


models = {
    'KNN': KNeighborsClassifier(n_neighbors=4),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    accuracy, f1, report = train_evaluate(model, X_train_scaled, X_test_scaled, y_train, y_test)
    accuracy_scores[name] = accuracy
    f1_scores[name] = f1
    classification_reports[name] = report
    
    accuracy, f1, report = train_evaluate(model, X_train_pca, X_test_pca, y_train, y_test)
    accuracy_scores[name + "_PCA"] = accuracy
    f1_scores[name + "_PCA"] = f1
    classification_reports[name + "_PCA"] = report

    accuracy, f1, report = train_evaluate(model, X_train_reduced, X_test_reduced, y_train, y_test)
    accuracy_scores[name + "_Reduced"] = accuracy
    f1_scores[name + "_Reduced"] = f1
    classification_reports[name + "_Reduced"] = report

max_accuracy_model = max(accuracy_scores, key=accuracy_scores.get)
max_accuracy = accuracy_scores[max_accuracy_model]

print(f"The model with the highest accuracy is {max_accuracy_model} with an accuracy of {max_accuracy}.")

for result, acc in accuracy_scores.items():
    print(f"{result} - Accuracy: {acc}, F1 Score: {f1_scores[result]}")
    #print(classification_reports[result])

plt.figure(figsize=(10, 8))
positions = np.arange(len(accuracy_scores))
plt.bar(positions - 0.4, list(accuracy_scores.values()), width=0.4, label='Accuracy', align='center', alpha=0.7)
plt.bar(positions, list(f1_scores.values()), width=0.4, label='F1 Score', align='center', alpha=0.7)
plt.xticks(positions, list(accuracy_scores.keys()), rotation=45)
plt.ylabel('Accuracy & F1')
plt.title('Performance Comparison of Different Models')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
plt.bar(positions, list(accuracy_scores.values()), width=0.4, label='Accuracy', align='center', alpha=0.7)
plt.xticks(positions, list(accuracy_scores.keys()), rotation=45)
plt.ylabel('Accuracy')
plt.title('Performance Comparison of Different Models')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
plt.bar(positions, list(f1_scores.values()), width=0.4, label='F1 Score', align='center', alpha=0.7)
plt.xticks(positions, list(accuracy_scores.keys()), rotation=45)
plt.ylabel('F1')
plt.title('Performance Comparison of Different Models')
plt.legend()
plt.show()


