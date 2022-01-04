import timeit
import pandas as pd
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


c=glasgow.groupby("Accident_Severity").Accident_Severity.agg(["count"])
print(c)


'''

glasgow2= glasgow[glasgow["Year"] >= 2010].copy()
print(glasgow2)
'''
#print(glasgow)
#print(glasgow.isna().sum())
'''
a = glasgow.groupby('Weather_Conditions').Weather_Conditions.agg(["count"]).reset_index().copy()
b = glasgow.groupby('Road_Surface_Conditions').Road_Surface_Conditions.agg(["count"]).reset_index().copy()
print(a)
print(b)
c=glasgow.groupby("Accident_Severity").Accident_Severity.agg(["count"])
print(c)
'''
toc = timeit.default_timer()
print(toc - tic)
