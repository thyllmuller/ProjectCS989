import timeit
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
tic = timeit.default_timer()


# "Time", <- maybe implement time in this

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
    ["Accident_Severity", "Number_of_Vehicles", "Number_of_Casualties", "Speed_limit", "Urban_or_Rural_Area",
     "Year"]].copy()

glasgow_x = glasgow.drop(["Accident_Severity"], axis=1)
glasgow_y = glasgow["Accident_Severity"]
#print(glasgow_y)
'''ADABOOST''' "Boosting-based Ensemble learning: sequential learning technique"
# base_estimator: It is a weak learner used to train the model.
# It uses DecisionTreeClassifier as default weak learner for training purpose.
# You can also specify different machine learning algorithms
# n_estimators: Number of weak learners to train iteratively until boosting is terminated.
# learning_rate: It contributes to the weights of weak learners. It uses 1 as a default value.
#   Reducing the learning rate will mean the weights will be increased or decreased to a small degree,
#   forcing the model train slower (but sometimes resulting in better performance scores).

x_train, x_test, y_train, y_test = train_test_split(glasgow_x, glasgow_y, test_size=0.2)  # 70% training and 30% test
abc = AdaBoostClassifier(n_estimators=1000, learning_rate=0.2)
# training model (weak, usually a decision tree):
model = abc.fit(x_train.values, y_train.values)
# using model on our dataset:
y_pred = model.predict(x_test.values)

# evaluating our model:
print("AdaBoost Accuracy:", metrics.accuracy_score(y_test, y_pred))

# some notes: (pros/cons)
# pros of using adaboost: simple, corrects mistakes of the weak classifier and improves accuracy by combining weak learners.
# adaboost is not likely to overfit either
# cons include that it is sensitive to noise (however, even with a noisy dataset this seems to still result in high accuracy!)

toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc-tic):.2f} seconds."))
