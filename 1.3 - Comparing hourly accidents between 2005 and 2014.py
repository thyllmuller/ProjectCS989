import pandas as pd
from matplotlib import pyplot as plt

import numpy as np
# Load the data in.

# Load the data in.
acc_05_07 = pd.read_csv("../accidents_2005_to_2007.csv", low_memory=False)
acc_09_11 = pd.read_csv("../accidents_2009_to_2011.csv", low_memory=False)
acc_12_14 = pd.read_csv("../accidents_2012_to_2014.csv", low_memory=False)

df1 = acc_05_07.drop_duplicates(
    subset=['Accident_Index', 'Date', 'LSOA_of_Accident_Location', 'Time', 'Longitude', 'Latitude'], keep='first')
df2 = acc_09_11.drop_duplicates(
    subset=['Accident_Index', 'Date', 'LSOA_of_Accident_Location', 'Time', 'Longitude', 'Latitude'], keep='first')
df3 = acc_12_14.drop_duplicates(
    subset=['Accident_Index', 'Date', 'LSOA_of_Accident_Location', 'Time', 'Longitude', 'Latitude'], keep='first')



'''PART1 SUMMARY STATISTICS'''
# print(acc_05_07.head())

'''
# x = dataset, y = year
def time_freq_glasgow(x, y):
    a = x[x["Year"] == y]
    b = a[a["Local_Authority_(Highway)"] == "S12000043"].copy()
    b["Time"] = pd.to_datetime(b["Time"], format="%H:%M").dt.hour
    counts = b.groupby('Time').Time.agg(["count"])
    label = str(f"Frequency of accidents per hour in Glasgow for the year of {y}.")
    fig, ax = plt.subplots()
    plt.style.use('grayscale')
    ax.plot(counts.loc[:, :])  # yes i know theres an error but i dont know how to fix this one
    ax.set(xlabel='24-Hour Time', ylabel='Frequency of Accidents')
    ax.set_title(label, fontsize=26)  # i know its cos of chained indexing
    xticks = np.arange(0, 24, 1)
    ax.set_xticks(xticks, minor=False)  # how i fix that is completely out of my depth --
    output_name = str(f"Frequency of Accidents in Glasgow {y}")
    fig.set_figheight(10)
    fig.set_figwidth(18)
    fig.savefig(output_name)
    plt.show()  # nvm fixed it with .copy() to duplicate the df
'''

#time_freq_glasgow(acc_12_14, 2014)




def time_freq_glasgow2x(x, y, x2, y2):
    a = x[x["Year"] == y]
    b = a[a["Local_Authority_(Highway)"] == "S12000043"].copy()
    b["Time"] = pd.to_datetime(b["Time"], format="%H:%M").dt.hour
    counts = b.groupby('Time').Time.agg(["count"])

    a2 = x2[x2["Year"] == y2]
    b2 = a2[a2["Local_Authority_(Highway)"] == "S12000043"].copy()
    b2["Time"] = pd.to_datetime(b2["Time"], format="%H:%M").dt.hour
    counts2 = b2.groupby("Time").Time.agg(["count"])

    label = str(f"Comparing Frequency of Accidents per Hour ({y} and {y2}).")
    plt.style.use("bmh")
    fig, ax = plt.subplots()
    ax.plot(counts.loc[:, :], label=f"{y}")
    ax.plot(counts2.loc[:, :], label=f"{y2}")
    plt.xlabel("Time (24-Hour)", fontsize=25)
    plt.ylabel("Number of Accidents", fontsize=25)
    ax.set_title(label, fontsize=32)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    xticks = np.arange(0, 24, 1)
    ax.set_xticks(xticks, minor=False)
    ax.legend()
    output_name = str(f"Comparing {y} with {y2}")
    fig.set_figheight(10)
    fig.set_figwidth(18)
    fig.savefig(output_name)
    plt.show()

time_freq_glasgow2x(df1, 2005, df3, 2014)
