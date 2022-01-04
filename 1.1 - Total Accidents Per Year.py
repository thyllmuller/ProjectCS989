import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
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


def all_accidents(a, b, c, ):
    df1x = a.append(b, ignore_index=True)
    df2x = df1x.append(c, ignore_index=True).copy()
    acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"]
    counts = acc_glasgow.groupby('Year').Year.agg(["count"])
    label = str("Comparing the Total Accident Count for each Year.")
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    ax.plot(counts)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Number of Accidents', fontsize=20)
    ax.set_title(label, fontsize=36, y=1, pad=10)
    xticks = np.arange(2005, 2015, 1)
    ax.set_xticks(xticks, minor=False)
    output_name = str("Comparing all years")
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.savefig(output_name)
    plt.show()


all_accidents(df1, df2, df3)
