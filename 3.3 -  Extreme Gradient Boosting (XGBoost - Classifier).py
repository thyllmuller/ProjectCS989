import timeit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier

tic = timeit.default_timer()


'''LOADING DATA AND CLEANING IT/SKIMMING IT DOWN (HAD TO DROP ALL USELESS COLUMNS (SOME HAD NA AS WELL)'''


def LoadData():
    acc_05_07 = pd.read_csv("../accidents_2005_to_2007.csv", low_memory=False)
    acc_09_11 = pd.read_csv("../accidents_2009_to_2011.csv", low_memory=False)
    acc_12_14 = pd.read_csv("../accidents_2012_to_2014.csv", low_memory=False)
    accidents = [acc_05_07, acc_09_11, acc_12_14]
    accidents_df = pd.concat(accidents)
    accidents_clean = accidents_df.drop_duplicates(
        subset=["Accident_Index", "Date", "LSOA_of_Accident_Location", "Time", "Longitude", "Latitude"], keep="first")
    final_glasgow = accidents_clean[accidents_clean["Local_Authority_(Highway)"] == "S12000043"].copy()
    return final_glasgow


glasgow = LoadData()
glasgow = glasgow[
    ["Accident_Severity", "Number_of_Vehicles", "Number_of_Casualties", "Speed_limit", "Year"]].copy()

glasgow_x = glasgow.drop(["Accident_Severity"], axis=1)
glasgow_y = glasgow["Accident_Severity"]

'''XGBOOST'''

# Creating split for training and testing
x_train, x_test, y_train, y_test = train_test_split(glasgow_x, glasgow_y, test_size=0.3)
D_train = xgb.DMatrix(x_train, label=y_train)
D_test = xgb.DMatrix(x_test, label=y_test)




'''ASCERTAINING WHICH CLASSIFIER TO USE:'''
#model = XGBClassifier()
#model.fit(x_train, y_train)
#print(model.objective)

'''MULTI:SOFTPROB'''

param = {
    'eta': 0.1,
    'max_depth': 3,
    'objective': 'multi:softprob',
    'num_class': 4}

steps = 20

'''Running XGBoost: Notes are at the borrom for params'''

xg_class = xgb.train(param, D_train, steps)
preds = xg_class.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])
precision = precision_score(y_test, best_preds, average='macro')
recall = recall_score(y_test, best_preds, average='macro')
accuracy = accuracy_score(y_test, best_preds)
print(str(f"Precision:{precision:.2f}"))
print(str(f"Recall:{recall:.2f}"))
print(str(f"Accuracy:{accuracy:.2f}"))


toc = timeit.default_timer()

print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))



xgb.plot_tree(xg_class,num_trees=2,rankdir='LR')
plt.show()


'''FOR XGBoost'''
# learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
# max_depth: determines how deeply each tree is allowed to grow during any boosting round.
# subsample: percentage of samples used per tree. Low value can lead to underfitting.
# colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
# n_estimators: number of trees you want to build.
# objective: determines the loss function to be used like reg:linear for regression problems (deprecated; now reg:squarederror),
#   reg:logistic for classification problems with only decision,
#   binary:logistic for classification problems with probability.
# XGBoost also supports regularization parameters to penalize models as they become more complex and
#   reduce them to simple (parsimonious) models.
# gamma: controls whether a given node will split based on the expected reduction in loss after the split.
#   A higher value leads to fewer splits. Supported only for tree-based learners.
# alpha: L1 regularization on leaf weights. A large value leads to more regularization.
# lambda: L2 regularization on leaf weights and is smoother than L1 regularization.


'''FOR k-fold validation:'''
# num_boost_round: denotes the number of trees you build (analogous to n_estimators)
# metrics: tells the evaluation metrics to be watched during CV
# as_pandas: to return the results in a pandas DataFrame.
# early_stopping_rounds: finishes training of the model early if the hold-out metric ("rmse" in our case)
#   does not improve for a given number of rounds.
