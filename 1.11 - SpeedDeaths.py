import timeit

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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


glasgowSD = glasgow[["Accident_Severity", "Speed_limit"]].copy()
glasgowSD2 = glasgowSD[glasgowSD["Accident_Severity"] == 1].copy()
print(glasgowSD2)
counts = glasgowSD2.groupby('Speed_limit').Speed_limit.agg(["count"])
print(counts)