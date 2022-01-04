import timeit
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

tic = timeit.default_timer()


def LoadData():
    acc_05_07 = pd.read_csv("../accidents_2005_to_2007.csv", low_memory=False)
    acc_09_11 = pd.read_csv("../accidents_2009_to_2011.csv", low_memory=False)
    acc_12_14 = pd.read_csv("../accidents_2012_to_2014.csv", low_memory=False)
    accidents = [acc_05_07, acc_09_11, acc_12_14]
    accidents_df = pd.concat(accidents)
    accidents_clean = accidents_df.drop_duplicates(
        subset=["Accident_Index", "Date", "LSOA_of_Accident_Location", "Time", "Longitude", "Latitude"], keep="first")
    glasgow_final = accidents_clean[accidents_clean["Local_Authority_(Highway)"] == "S12000043"].copy()
    return glasgow_final


glasgow = LoadData()
glasgow = glasgow[
    ["Accident_Severity", "Number_of_Vehicles", "Number_of_Casualties", "Speed_limit", "Urban_or_Rural_Area",
     "Year"]].copy()
glasgow_x = glasgow.drop(["Accident_Severity"], axis=1)
glasgow_y = glasgow["Accident_Severity"]

'''LOGISTIC REGRESSION'''

x_train, x_test, y_train, y_test = train_test_split(glasgow_x, glasgow_y, test_size=0.3)
logreg = LogisticRegression(solver='lbfgs', max_iter=200)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg.score(x_test, y_test)))

toc = timeit.default_timer()

print("Time taken to complete task:", "{:.2f}".format(toc-tic), "seconds.")
