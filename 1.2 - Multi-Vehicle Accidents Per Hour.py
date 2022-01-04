import pandas as pd
from matplotlib import pyplot as plt

import numpy as np

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


def multi_car_collisions(a, b, c):
    df1x = a.append(b, ignore_index=True)
    df2x = df1x.append(c, ignore_index=True).copy()
    #print(df2x.keys()) # too lazy to open the excel file and check columns
    acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"].copy()
    acc_glasgow["Time"] = pd.to_datetime(acc_glasgow["Time"], format="%H:%M").dt.hour.copy()
    counts = acc_glasgow.groupby(["Number_of_Vehicles", "Time"]).Number_of_Vehicles.agg(["count"]).reset_index().copy()
    # print(counts) checks
    ndf1 = counts[counts["Number_of_Vehicles"] <= 1]
    ndf2 = counts[counts["Number_of_Vehicles"] == 2]
    ndf3 = counts[counts["Number_of_Vehicles"] == 3]
    ndf4 = counts[counts["Number_of_Vehicles"] == 4]
    ndf5 = counts[counts["Number_of_Vehicles"] >= 5]
    ndf1.set_index("Number_of_Vehicles", inplace=True)
    ndf2.set_index("Number_of_Vehicles", inplace=True)
    ndf3.set_index("Number_of_Vehicles", inplace=True)
    ndf4.set_index("Number_of_Vehicles", inplace=True)
    ndf5.set_index("Number_of_Vehicles", inplace=True)
    print(ndf2)
    print(ndf3)
    print(ndf4)
    print(ndf5)
    label = str("Investigating the Frequency of Multi-Vehicle Accidents for every Hour.")
    fig, ax = plt.subplots()
    w = 4
    #ax.bar(ndf1["Time"], ndf1["count"], width=w / 5)
    ax.bar(ndf2["Time"], ndf2["count"], width=w / 5)
    ax.bar(ndf3["Time"], ndf3["count"], width=w / 5)
    ax.bar(ndf4["Time"], ndf4["count"], width=w / 5)
    ax.bar(ndf5["Time"], ndf5["count"], width=w / 5)
    plt.xlabel('Hour', fontsize=30)
    plt.ylabel('Number of Accidents', fontsize=30)
    ax.set_title(label, fontsize=35)
    xticks = np.arange(0, 24, 1)
    ax.set_xticks(xticks, minor=False)
    ax.legend(["2 Vehicles", "3 Vehicles", "4 Vehicles", "5+ Vehicles"])
    output_name = str("Comparing number of vehicles per hour")
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.savefig(output_name)
    plt.show()


multi_car_collisions(df1, df2, df3)
