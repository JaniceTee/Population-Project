# INSTALL AND IMPORT ALL NECESSARY LIBRARIES.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# DATA PREPARATION/WRANGLING
df = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\555 Project\Janice Tagoe_census-historic-population-borough.csv")

print('Data Shape: ', df.shape)
print('\nData Types: ', '\n', df.dtypes)
print('\n', df.head())
print('\nCount of Null Values: ', '\n', df.isnull().sum())
print('\nChecking for Duplicate Records: ', '\n', df.duplicated())

# DATA EXPLORATION
# 1.Visualization of Target Variable (Forecasted Growth)
forecasted_growth = df['growth'].value_counts()
print('\nForecasted Growth:', '\n', forecasted_growth)
sns.countplot(x='growth', data=df, palette='hls')
plt.title('Forecasted Growth: Up(1)/Down(0)')
plt.show()

# 2.Ratio of Forecasted Growth (Up/Down)
count_downward_growth = len(df[df['growth'] == 0])
count_upward_growth = len(df[df['growth'] == 1])
pct_of_downward_growth = count_downward_growth/(count_downward_growth+count_upward_growth)
print('\npercentage of boroughs experiencing a decline in population change is', pct_of_downward_growth*100)
pct_of_upward_growth = count_upward_growth/(count_downward_growth+count_upward_growth)
print('percentage of boroughs experiencing an upward growth in population change is', pct_of_upward_growth*100)

# 3.Description of Predictors
predictors = df[['Pop_1991', 'Pop_2001', 'Pop_2011', 'Pop_2021']]
print('\nDescription of Predictors: ', '\n', predictors.describe())

# 4.Comparing the population values of the last four census on a single graph.
df = df.rename(columns={'Area Name': 'names'})
x_coordinates = df.names
y1_coordinates = df.Pop_1991
y2_coordinates = df.Pop_2001
y3_coordinates = df.Pop_2011
y4_coordinates = df.Pop_2021
plt.plot(x_coordinates, y1_coordinates, label="1991")
plt.plot(x_coordinates, y2_coordinates, label="2001")
plt.plot(x_coordinates, y3_coordinates, label="2011")
plt.plot(x_coordinates, y3_coordinates, label="2021")
plt.xticks(x_coordinates[::1],  rotation='vertical')
plt.margins(0.2)
plt.title('Population of London Boroughs over Last Four Decades')
plt.xlabel('London Boroughs')
plt.ylabel('Population')
plt.tight_layout()
plt.legend()
plt.show()

# 5.Correlation Matrix and Heatmap of Predictors
print('\nCorrelation Matrix: ', '\n', predictors.corr())
sns.heatmap(predictors.corr(), cmap="YlGnBu", annot=True)
plt.title('Census Correlation Heatmap')

# DEVELOP AND EVALUATE MODELS
x = df[['Pop_1991',	'Pop_2001',	'Pop_2011',	'Pop_2021']].values
y = df['growth']

# SPLIT THE DATA INTO THE TRAIN AND TEST SETS.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=0)

# 1.LOGISTIC REGRESSION MODEL.
lr = LogisticRegression(solver='liblinear', max_iter=400).fit(x_train, y_train)
lr_prediction = lr.predict(x_test)
lr_sq = lr.score(x, y)
print('\nLOGISTIC REGRESSION MODEL: ')
print(f'Coefficient of Determination: {lr_sq}')
print('Accuracy score: ', accuracy_score(y_test, lr_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, lr_prediction))
print('\nFrom Test Data: ')
for index in range(len(lr_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', lr_prediction[index])

# PREDICT FROM DATASET
lr_DataToPredict = np.array([[155698, 172330,	186990,	195200], [198916, 244867, 288283, 307700],
                             [164868, 179767, 190146, 209600], [153255, 196083, 254096, 310300],
                             [203343, 218335, 258249, 278400], [239162, 260379, 306995, 327500],
                             [177743, 181284, 219396, 204300], [2343133, 2766065, 3231901, 3332345],
                             [4050435, 4405992, 4942040, 5045687], [6393568, 7172057,	8173941, 8384152]])
lr_pred = lr.predict(lr_DataToPredict)
print('\nPredicted results: ')
for index in range(len(lr_pred)):
    print('\t', lr_DataToPredict[index], '\t\t\tPredicted: ', lr_pred[index])

# EVALUATE THE LOGISTIC REGRESSION MODEL
print('\n', 'Classification Report for Logistic Regression Model: ', '\n', classification_report(y_test, lr_prediction))

# 2.K-NEAREST NEIGHBORS (KNN) MODEL
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
knn_prediction = knn.predict(x_test)
knn_sq = knn.score(x, y)
print('KNN MODEL:')
print(f'Coefficient of Determination: {knn_sq}')
print('Accuracy score: ', accuracy_score(y_test, knn_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, knn_prediction))
print('\nFrom Test Data: ')
for index in range(len(knn_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', knn_prediction[index])

# PREDICT FROM DATASET
knn_DataToPredict = np.array([[155698, 172330,	186990,	195200], [198916, 244867, 288283, 307700],
                              [164868, 179767, 190146, 209600], [153255, 196083, 254096, 310300],
                              [203343, 218335, 258249, 278400], [239162, 260379, 306995, 327500],
                              [177743, 181284, 219396, 204300], [2343133, 2766065, 3231901, 3332345],
                              [4050435, 4405992, 4942040, 5045687], [6393568, 7172057,	8173941, 8384152]])
knn_pred = knn.predict(knn_DataToPredict)
print('\nPredicted results: ')
for index in range(len(knn_pred)):
    print('\t', knn_DataToPredict[index], '\t\t\tPredicted: ', knn_pred[index])

# EVALUATE THE KNN MODEL
print('\n', 'Classification Report for KNN Model: ', '\n', classification_report(y_test, knn_prediction))

# 3.NAIVE BAYES MODEL
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_prediction = nb.predict(x_test)
nb_sq = nb.score(x, y)
print('\nNAIVE BAYES MODEL:')
print(f'Coefficient of Determination: {nb_sq}')
print('Accuracy score: ', accuracy_score(y_test, nb_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, nb_prediction))
print('\nFrom Test Data: ')
for index in range(len(nb_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', nb_prediction[index])

# PREDICT FROM DATASET
nb_DataToPredict = np.array([[155698, 172330,	186990,	195200], [198916, 244867, 288283, 307700],
                             [164868, 179767, 190146, 209600], [153255, 196083, 254096, 310300],
                             [203343, 218335, 258249, 278400], [239162, 260379, 306995, 327500],
                             [177743, 181284, 219396, 204300], [2343133, 2766065, 3231901, 3332345],
                             [4050435, 4405992, 4942040, 5045687], [6393568, 7172057,	8173941, 8384152]])
nb_pred = nb.predict(nb_DataToPredict)
print('\nPredicted results: ')
for index in range(len(nb_pred)):
    print('\t', nb_DataToPredict[index], '\t\t\tPredicted: ', nb_pred[index])

# EVALUATE THE NAIVE BAYES MODEL
print('\n', 'Classification Report for Naive Bayes Model: ', '\n', classification_report(y_test, nb_prediction))

# 4.DECISION TREE CLASSIFIER MODEL
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_prediction = dtc.predict(x_test)
dtc_sq = dtc.score(x, y)
print('\nDECISION TREE CLASSIFIER MODEL:')
print(f'Coefficient of Determination: {dtc_sq}')
print('Accuracy score: ', accuracy_score(y_test, dtc_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, dtc_prediction))
print('\nFrom Test Data: ')
for index in range(len(dtc_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', dtc_prediction[index])

# PREDICT FROM DATASET
dtc_DataToPredict = np.array([[155698, 172330,	186990,	195200], [198916, 244867, 288283, 307700],
                              [164868, 179767, 190146, 209600], [153255, 196083, 254096, 310300],
                              [203343, 218335, 258249, 278400], [239162, 260379, 306995, 327500],
                              [177743, 181284, 219396, 204300], [2343133, 2766065, 3231901, 3332345],
                              [4050435, 4405992, 4942040, 5045687], [6393568, 7172057,	8173941, 8384152]])
dtc_pred = dtc.predict(dtc_DataToPredict)
print('\nPredicted results: ')
for index in range(len(dtc_pred)):
    print('\t', dtc_DataToPredict[index], '\t\t\tPredicted: ', dtc_pred[index])

# EVALUATE THE DECISION TREE MODEL
print('\n', 'Classification Report for Decision Tree Model : ', '\n', classification_report(y_test, dtc_prediction))

# 5.RANDOM FOREST MODEL
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
rf_sq = rf.score(x, y)
print('\nRANDOM FOREST REGRESSION MODEL:')
print(f'Coefficient of Determination: {rf_sq}')
print('Accuracy score: ', accuracy_score(y_test, rf_prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, rf_prediction))
print('\nFrom Test Data: ')
for index in range(len(rf_prediction)):
    print('Index: ', index, 'Actual: ', y[index], 'Predicted: ', rf_prediction[index])

# PREDICT FROM DATASET
rf_DataToPredict = np.array([[155698, 172330,	186990,	195200], [198916, 244867, 288283, 307700],
                             [164868, 179767, 190146, 209600], [153255, 196083, 254096, 310300],
                             [203343, 218335, 258249, 278400], [239162, 260379, 306995, 327500],
                             [177743, 181284, 219396, 204300], [2343133, 2766065, 3231901, 3332345],
                             [4050435, 4405992, 4942040, 5045687], [6393568, 7172057, 8173941, 8384152]])
rf_pred = rf.predict(rf_DataToPredict)
print('\nPredicted results: ')
for index in range(len(rf_pred)):
    print('\t', rf_DataToPredict[index], '\t\t\tPredicted: ', rf_pred[index])

# EVALUATE THE RANDOM FOREST MODEL
print('\n', 'Classification Report for Random Forest Model: ', '\n', classification_report(y_test, rf_prediction))

# LOGISTIC REGRESSION ROC CURVE
logit_roc_auc = roc_auc_score(y_test, lr.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# KNN ROC CURVE
logit_roc_auc = roc_auc_score(y_test, knn.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='K-Nearest Neighbor AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for K-Nearest Neighbor')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# NAIVE BAYES ROC CURVE
logit_roc_auc = roc_auc_score(y_test, nb.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, nb.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Naive Bayes AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# DECISION TREE ROC CURVE
logit_roc_auc = roc_auc_score(y_test, dtc.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, dtc.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for Decision Tree Classifier')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# RANDOM FOREST ROC CURVE
logit_roc_auc = roc_auc_score(y_test, rf.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.05, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve for Random Forest Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# COMPARING MODELS
models = ['Logistic Regression', 'KNN', 'Naive Bayes', 'Decision Tree', 'Random Forest']
acc_score = [accuracy_score(y_test, lr_prediction), accuracy_score(y_test, knn_prediction), accuracy_score(y_test, nb_prediction),
             accuracy_score(y_test, dtc_prediction), accuracy_score(y_test, rf_prediction)]
r_square = [lr_sq, knn_sq, nb_sq, dtc_sq, rf_sq]
x_axis = np.arange(len(models))
# Multi bar Chart
plt.bar(x_axis - 0.2, acc_score, width=0.4, label='Accuracy Score')
plt.bar(x_axis + 0.2, r_square, width=0.4, label='R Square Values')
plt.xticks(x_axis, models)
# Add legend
plt.legend()
# Display
plt.show()




