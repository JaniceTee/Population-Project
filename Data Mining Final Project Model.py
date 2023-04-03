# Install and import necessary libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from firthlogist import load_endometrial
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\princ\Downloads\census-historic-population-borough_London (1).csv")

x = df[['Pop_1991',	'Pop_2001',	'Pop_2011',	'Pop_2021']].values
y = df['growth']

# Split the data into the train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=0)

# Fit the Logistic Regression Model.
model = LogisticRegression(solver='liblinear', max_iter=400).fit(x_train, y_train)
prediction = model.predict(x_test)

print('Accuracy score: ', accuracy_score(y_test, prediction))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, prediction))

for index in range(len(prediction)):
    print('Actual: ', y[index], 'Predicted: ', prediction[index])

# Predict the population growth for specific boroughs for a specific census year. (Pop_1981 for Enfield,
# Pop_1871 for Hammersmith and Fulham, Pop_1901 for Bexley, Pop_1921 for Newham, Pop_1939 for Lambeth,
# Pop_1951 for Tower Hamlets, Pop_1911 for Westminster, Pop_1939 for Havering,
# Pop_1939 for Islington, and Pop_1841 for Kensington and Chelsea respectively.)
DataToPredict = np.array([[272000,	288000,	273857,	268000], [18000, 23000, 30000, 40000],
                          [15000, 22000, 29000, 37000], [142000, 240000, 366000, 427000],
                          [384000, 408000, 419000, 421000], [570000, 529000, 489000, 419000],
                          [524000, 513000, 481000, 460000], [26000,	33000,	38000,	77000],
                          [436000, 415000, 407000, 392000], [18000, 25000, 35000, 46000]])
pred = model.predict(DataToPredict)

print("\nPredicted results")
for i in range(len(pred)):
    print('\t', DataToPredict[i], '\t', pred[i])

# EVALUATE THE MODEL
print('\n', 'Classification Report: ', '\n', classification_report(y_test, prediction))

# ROC Curve
logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# MODEL INTERPRETATION
print('\nModel Coefficient')
print(model.coef_)
print('\nModel intercept')
print(model.intercept_)

x, y, feature_names = load_endometrial()
log_reg = sm.Logit(y, x).fit()
print(log_reg.summary2())

