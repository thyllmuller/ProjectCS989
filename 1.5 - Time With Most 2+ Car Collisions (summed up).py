
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

def multi_car_collisions(a,b,c):
    df1x = a.append(b, ignore_index=True)
    df2x = df1x.append(c, ignore_index=True).copy()
    acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"].copy()
    acc_glasgow_filt = acc_glasgow[acc_glasgow["Number_of_Vehicles"] >= 2].copy()
    acc_glasgow_filt["Time"] = pd.to_datetime(acc_glasgow_filt["Time"], format="%H:%M").dt.hour.copy()
    most_freq_time = acc_glasgow_filt.groupby(["Time"]).Time.agg(["count"]).reset_index()
    print(most_freq_time)
    label = str("Investigating the frequency of multi-car (2+) accidents for every hour (2005-2014).")
    fig, ax = plt.subplots()
    ax.bar(most_freq_time["Time"], most_freq_time["count"])
    ax.set(xlabel='Hour', ylabel='Number of Accidents')
    ax.set_title(label, fontsize=26)
    xticks = np.arange(0, 24, 1)
    ax.set_xticks(xticks, minor=False)
    for i, val in enumerate(most_freq_time["count"]):
        plt.text(i - 0.25, val - 20, str(f"{val}"), color='white', fontweight='bold')
    output_name = str("Time With Most 2+ Car Collisions")
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.savefig(output_name)
    plt.show()

multi_car_collisions(df1,df2,df3)