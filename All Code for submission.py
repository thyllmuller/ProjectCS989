import pandas as pd
import sns as sns
import numpy as np
import timeit
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import folium
from geopy.distance import great_circle
from sklearn import metrics
from sklearn.cluster import DBSCAN as dbscan
from sklearn.datasets import make_blobs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

'''1.1 Total Accidents Per Year Graph'''

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


'''1.2 Multi-Vehicle Accidents Per Hour'''


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



'''1.3 Comparing Hourly Accidents Between 2005 and 2014'''
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


'''1.4 Time with most single car collisions'''


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


def single_car_collisions(a, b, c):
    # plt.style.use("seaborn-dark-palette")
    df1x = a.append(b, ignore_index=True)
    df2x = df1x.append(c, ignore_index=True).copy()
    acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"].copy()
    acc_glasgow_filt = acc_glasgow[acc_glasgow["Number_of_Vehicles"] == 1].copy()
    acc_glasgow_filt["Time"] = pd.to_datetime(acc_glasgow_filt["Time"], format="%H:%M").dt.hour.copy()
    most_freq_time = acc_glasgow_filt.groupby(["Time"]).Time.agg(["count"]).reset_index()
    print(most_freq_time)
    label = str("Investigating the Frequency of Single-Vehicle Accidents for every Hour.")
    fig, ax = plt.subplots()
    ax.bar(most_freq_time["Time"], most_freq_time["count"])
    ax.set(xlabel='Hour', ylabel='Number of Accidents')
    plt.xlabel("Time (24-Hour)", fontsize=25)
    plt.ylabel("Number of Accidents", fontsize=25)
    ax.set_title(label, fontsize=32)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    xticks = np.arange(0, 24, 1)
    ax.set_xticks(xticks, minor=False)
    for i, val in enumerate(most_freq_time["count"]):
        plt.text(i - 0.25, val - 10, str(f"{val}"), color='white', fontweight='bold')
    output_name = str("Time With Most Single Car Collisions")
    fig.set_figheight(10)
    fig.set_figwidth(20)
    fig.savefig(output_name)
    plt.show()


single_car_collisions(df1, df2, df3)

'''1.5 Time with most 2+ car collisins (summed)'''
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

'''1.6 Correlation heatmap'''

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

labels = ["Accident Severity", "Number of Vehicles",
                                "Number of Casualties", "Day of Week", "Speed limit",
                                "Weather Conditions", "Road Surface Conditions",
                                "Urban or Rural Area", "Year"]

df1x = df1.append(df2, ignore_index=True)
df2x = df1x.append(df3, ignore_index=True).copy()
acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"].copy()
acc_glasgow_corr = acc_glasgow[["Accident_Severity", "Number_of_Vehicles",
                                "Number_of_Casualties", "Day_of_Week", "Speed_limit",
                                "Weather_Conditions", "Road_Surface_Conditions",
                                "Urban_or_Rural_Area", "Year"]]
pd.set_option('display.max_columns', None)
#print(acc_glasgow_corr)
acc_glasgow_corr_added = acc_glasgow_corr.replace({
    'Weather_Conditions': {
        'Fine with high winds': 0,
        'Fine without high winds': 1,
        'Fog or mist': 2,
        'Other': 3,
        'Raining with high winds': 4,
        'Raining without high winds': 5,
        'Snowing with high winds': 6,
        'Snowing without high winds': 7,
        'Unknown': 8
    },
    'Road_Surface_Conditions': {
        'Dry': 0,
        'Flood (Over 3cm of water)': 1,
        'Frost/Ice': 2,
        'Snow': 3,
        'Wet/Damp': 4
    }}).copy()
#print(acc_glasgow_corr_added)


corr = acc_glasgow_corr_added.corr()
fig, ax = plt.subplots(figsize=(14,10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0,
             cmap=(sns.diverging_palette(250, 20, s=100, l=50, n=230, center="dark")),  linewidths=1, linecolor="white",
             square=True, annot=True, cbar_kws={"orientation": "horizontal", "shrink": .60})
ax.set_xticklabels(labels, rotation=45, horizontalalignment="right")
ax.set_yticklabels(labels)
ax.set_title("Correlation of all Variables in the Filtered Dataset", fontsize=30, y=1, pad=10)
output_name = str("Correlation Heatmap")
plt.tight_layout()
fig.savefig(output_name)
plt.show()


'''1.7 Ranked Correlation Heatmap'''

tic = timeit.default_timer()
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

df1x = df1.append(df2, ignore_index=True)
df2x = df1x.append(df3, ignore_index=True).copy()

labels = ["Accident Severity", "Number of Vehicles",
                                "Number of Casualties", "Day of Week", "Speed limit",
                                "Weather Conditions", "Road Surface Conditions",
                                "Urban or Rural Area", "Year"]


acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"].copy()
acc_glasgow_corr = acc_glasgow[["Accident_Severity", "Number_of_Vehicles",
                                "Number_of_Casualties", "Day_of_Week", "Speed_limit",
                                "Weather_Conditions", "Road_Surface_Conditions",
                                "Urban_or_Rural_Area", "Year"]]
pd.set_option('display.max_columns', None)
#print(acc_glasgow_corr)
acc_glasgow_corr_added = acc_glasgow_corr.replace({
    'Weather_Conditions': {
        'Fine with high winds': 0,
        'Fine without high winds': 1,
        'Fog or mist': 2,
        'Other': 3,
        'Raining with high winds': 4,
        'Raining without high winds': 5,
        'Snowing with high winds': 6,
        'Snowing without high winds': 7,
        'Unknown': 8
    },
    'Road_Surface_Conditions': {
        'Dry': 0,
        'Flood (Over 3cm of water)': 1,
        'Frost/Ice': 2,
        'Snow': 3,
        'Wet/Damp': 4
    }}).copy()

fig, ax = plt.subplots(figsize=(8,10))
sns.heatmap(acc_glasgow_corr_added.corr()[['Speed_limit']].sort_values(by='Speed_limit', ascending=False),
            vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor='white',
            cmap=(sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark")),
            square=True, annot=True, cbar=False)
ax.tick_params(right=True, top=False, labelright=True, labeltop=False, left=False, labelleft=False, bottom=False, labelbottom=True)
ax.set_yticklabels(labels, rotation=0, horizontalalignment='left', fontsize=16)
ax.set_xlabel('')
ax.set(xticklabels=[])
output_name = str("Ranked Correlation Heatmap")
plt.tight_layout()
fig.savefig(output_name)
plt.show()
toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc-tic):.2f} seconds."))


'''1.8 PairPlot - Newer code in section 1.9'''


tic = timeit.default_timer()

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

df1x = df1.append(df2, ignore_index=True)
df2x = df1x.append(df3, ignore_index=True).copy()
acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"].copy()
acc_glasgow_pair = acc_glasgow[["Accident_Severity",
                                "Number_of_Vehicles",
                                "Number_of_Casualties",
                                "Speed_limit",
                                "Weather_Conditions",
                                "Road_Surface_Conditions"]]
acc_glasgow_pair2 = acc_glasgow_pair[acc_glasgow_pair["Number_of_Casualties"] <= 10].copy()
labels = {"Accident_Severity": "Accident Severity",
          "Number_of_Vehicles": "Number of Vehicles",
          "Number_of_Casualties": "Number of Casualties",
          "Weather_Conditions": "Weather Conditions",
          "Road_Surface_Conditions": "Road Surface Conditions"}

acc_glasgow_pair3 = acc_glasgow_pair2.replace({
    'Weather_Conditions': {
        'Fine with high winds': 0,
        'Fine without high winds': 1,
        'Fog or mist': 2,
        'Other': 3,
        'Raining with high winds': 4,
        'Raining without high winds': 5,
        'Snowing with high winds': 6,
        'Snowing without high winds': 7,
        'Unknown': 8
    },
    'Road_Surface_Conditions': {
        'Dry': 0,
        'Flood (Over 3cm of water)': 1,
        'Frost/Ice': 2,
        'Snow': 3,
        'Wet/Damp': 4
    }}).copy()
plt.rcParams["axes.labelsize"] = 25
sns.set_style("darkgrid")
plot = sns.pairplot(acc_glasgow_pair3, hue='Speed_limit',
                    diag_kind="auto", height=4, aspect=10 / 5, kind="kde")
for i in range(5):
    for j in range(5):
        xlabel = plot.axes[i][j].get_xlabel()
        ylabel = plot.axes[i][j].get_ylabel()
        if xlabel in labels.keys():
            plot.axes[i][j].set_xlabel(labels[xlabel])
        if ylabel in labels.keys():
            plot.axes[i][j].set_ylabel(labels[ylabel])
output_name = str("PairPlot")
sns.set(font_scale=4)
plot.savefig(output_name)
plt.tight_layout()
plt.show()




toc = timeit.default_timer()

print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))

'''1.9 New Pairplot (condensed)'''


tic = timeit.default_timer()

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

df1x = df1.append(df2, ignore_index=True)
df2x = df1x.append(df3, ignore_index=True).copy()
acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"].copy()
acc_glasgow_pair = acc_glasgow[["Accident_Severity",
                                "Number_of_Vehicles",
                                "Number_of_Casualties",
                                "Speed_limit",
                                "Weather_Conditions",
                                "Road_Surface_Conditions"]]
acc_glasgow_pair2 = acc_glasgow_pair[acc_glasgow_pair["Number_of_Casualties"] <= 10].copy()
labels = {"Number_of_Vehicles": "Number of Vehicles",
          "Number_of_Casualties": "Number of Casualties",
          "Road_Surface_Conditions": "Road Surface Conditions"}

acc_glasgow_pair3 = acc_glasgow_pair2.replace({
    'Weather_Conditions': {
        'Fine with high winds': 0,
        'Fine without high winds': 1,
        'Fog or mist': 2,
        'Other': 3,
        'Raining with high winds': 4,
        'Raining without high winds': 5,
        'Snowing with high winds': 6,
        'Snowing without high winds': 7,
        'Unknown': 8
    },
    'Road_Surface_Conditions': {
        'Dry': 0,
        'Flood (Over 3cm of water)': 1,
        'Frost/Ice': 2,
        'Snow': 3,
        'Wet/Damp': 4
    }}).copy()


acc_glasgow_pair4 = acc_glasgow_pair3[["Number_of_Vehicles",
                                "Number_of_Casualties",
                                "Speed_limit",
                                "Road_Surface_Conditions"]]


plt.rcParams["axes.labelsize"] = 20
sns.set_style("darkgrid")
plot = sns.pairplot(acc_glasgow_pair4, hue='Speed_limit',
                    diag_kind="auto", height=4, aspect=10 / 5, kind="kde")
for i in range(3):
    for j in range(3):
        xlabel = plot.axes[i][j].get_xlabel()
        ylabel = plot.axes[i][j].get_ylabel()
        if xlabel in labels.keys():
            plot.axes[i][j].set_xlabel(labels[xlabel])
        if ylabel in labels.keys():
            plot.axes[i][j].set_ylabel(labels[ylabel])
output_name = str("PairPlotNew")
sns.set(font_scale=4)
plot.savefig(output_name)
plt.tight_layout()
plt.show()




toc = timeit.default_timer()

print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))

'''1.10 New Road and Weather Conditions'''

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

labels = ["Accident Severity", "Number of Vehicles",
          "Number of Casualties", "Day of Week", "Speed limit",
          "Weather Conditions", "Road Surface Conditions",
          "Urban or Rural Area", "Year"]

df1x = df1.append(df2, ignore_index=True)
df2x = df1x.append(df3, ignore_index=True).copy()
acc_glasgow = df2x[df2x["Local_Authority_(Highway)"] == "S12000043"].copy()
acc_glasgow_corr = acc_glasgow[["Accident_Severity", "Number_of_Vehicles",
                                "Number_of_Casualties", "Day_of_Week", "Speed_limit",
                                "Weather_Conditions", "Road_Surface_Conditions",
                                "Urban_or_Rural_Area", "Year"]]
pd.set_option('display.max_columns', None)
acc_glasgow_corr_added = acc_glasgow_corr.replace({
    'Weather_Conditions': {
        'Fine with high winds': 0,
        'Fine without high winds': 1,
        'Fog or mist': 2,
        'Other': 3,
        'Raining with high winds': 4,
        'Raining without high winds': 5,
        'Snowing with high winds': 6,
        'Snowing without high winds': 7,
        'Unknown': 8
    },
    'Road_Surface_Conditions': {
        'Dry': 0,
        'Flood (Over 3cm of water)': 1,
        'Frost/Ice': 2,
        'Snow': 3,
        'Wet/Damp': 4
    }}).copy()
final_glasgow = acc_glasgow_corr_added.copy()
final_glasgow_2 = final_glasgow[final_glasgow["Number_of_Casualties"] <= 50].copy()
final_glasgow_3=final_glasgow_2.groupby(['Accident_Severity', 'Road_Surface_Conditions']).Road_Surface_Conditions.agg(["count"]).reset_index()
print(final_glasgow_3)
sns.set_style("whitegrid")
fig, ax = plt.subplots()
sns.barplot(x=final_glasgow_3["Road_Surface_Conditions"], y=final_glasgow_3["count"],
            linewidth=1)
plt.xlabel("Road Surface Conditions", fontsize=25)
plt.ylabel("Number of Casualties", fontsize=25)
label = str(f"Accident Severity Distribution based on Road Surface- and Weather Conditions.")
ax.set_title(label, fontsize=28)
ax.tick_params(axis="both", which="major", labelsize=15)
ax.tick_params(axis="both", which="minor", labelsize=10)
# labelsy = ["Fatal", "Serious", "Slight"]
# ax.set_yticklabels(labelsy,rotation=55)
labelsx = ['Dry',
           'Flood (Over 3cm of water)',
           'Frost/Ice',
           'Snow',
           'Wet/Damp']
#ax.set_xticklabels(labelsx)

# ax.legend(handles=ax.legend_.legendHandles, labels=['Fine with high winds',
#           'Fine without high winds',
#           'Fog or mist',
#           'Other',
#           'Raining with high winds',
#           'Raining without high winds',
#           'Snowing with high winds',
#           'Snowing without high winds',
#           'Unknown'],prop={'size': 20})
output_name = str(f"Road and Weather Conditions Violin Plot")
fig.set_figheight(10)
fig.set_figwidth(20)
plt.tight_layout()
fig.savefig(output_name)
plt.show()

'''UNSUPERVISED LEARNING'''
'''2.0 KMeans'''


tic = timeit.default_timer()


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
glasgow2 = glasgow[["Latitude", "Longitude"]].copy()

# THE ELBOW CURVE
K_clusters = range(1, 10)

kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = glasgow2[['Latitude']]
X_axis = glasgow2[['Longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
plt.plot(K_clusters, score)
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.title("Elbow Curve")
output_name = str("kmeanselbow")
plt.savefig(output_name)
plt.show()

#THE KMEANS
#need to flatten plane to 2d with PCA(2)
pca = PCA(2)
df = pca.fit_transform(glasgow2)
kmeans = KMeans(n_clusters=4)
label = kmeans.fit_predict(df)
u_labels = np.unique(label)
centroids = kmeans.cluster_centers_
for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color="k")
plt.legend()
output_name = str("kmeansGLA")
plt.savefig(output_name)
plt.show()

toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))

'''2.1 DBSCAN'''

tic = timeit.default_timer()

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
'''UNSUPERVISED LEARNING: DBSCAN (Distance Based Clustering)'''


# I want to define my function as taking 3 variables, a and b and c.
# a = DF in question
# b = min_samples (default = 5)
# c = distance between accidents
def func_unsupervised_dbscan_all_years(a, b, c):
    print("Just so that my comment (in the code) below this is yellow \n")

    '''STAGE 1:SUBSETTING MY DATA FOR GLASGOW AND PER YEAR ONLY'''


    acc_glasgow_year = a
    #only for the one question below:
    #acc_glasgow_year = a[a["Number_of_Vehicles"] == 1]



    '''STAGE 1.2: DEFINING THE GEOPY FUNCTION'''

    # using the GeoPy package we can have access to the great_circle function which
    # lets me run more accurate analysis of the distance between two points (since it is not specified otherwise)
    # while taking into account the curvature of the earth

    def function_geopy_gc(x, y):
        lat1, long1 = x[0], x[1]
        lat2, long2 = y[0], y[1]
        distance = great_circle((lat1, long1), (lat2, long2)).meters
        return distance

    '''STAGE 2: DEFINING THE PARAMETERS FOR DBSCAN AND FORMING CLUSTERS'''
    dist_between_accidents = c  # distance in meters to use with dbscan
    # min_samples = The number of samples (or total weight) in a neighborhood for a
    # point to be considered as a core point. This includes the point itself.
    acc_glasgow_year_dbscan = acc_glasgow_year
    location = acc_glasgow_year_dbscan[["Latitude", "Longitude"]]
    dbs = dbscan(eps=dist_between_accidents, min_samples=b, metric=function_geopy_gc).fit(location)
    labels = dbs.labels_
    unique_labels = np.unique(dbs.labels_)
    print(unique_labels)  # I left this in as a check, I can make sure code is running correctly and changing clusters.
    acc_glasgow_year_dbscan["Cluster"] = labels
    # also important to note that any accident that was not in a cluster was given the value -1
    # this means it won't appear on the map we are going to form but could make a map with outliers later

    '''METRICS'''
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    X, labels_true = make_blobs(n_samples=len(acc_glasgow_year_dbscan["Latitude"]), centers=location, cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)
    print(str(f"Estimated number of clusters: {n_clusters_}"))
    print(str(f"Estimated number of noise points: {n_noise_}"))
    print(str(f"Homogeneity: {(metrics.homogeneity_score(labels_true, labels)):.3f}"))
    print(str(f"Completeness: {(metrics.completeness_score(labels_true, labels)):.3f}"))
    print(str(f"V-measure: {(metrics.v_measure_score(labels_true, labels)):.3f}"))
    print(str(f"Adjusted Rand Index: {(metrics.adjusted_rand_score(labels_true, labels)):.3f}"))
    print(str(f"Adjusted Mutual Information: {(metrics.adjusted_mutual_info_score(labels_true, labels)):.3f}"))
    print(str(f"Silhouette Coefficient: {(metrics.silhouette_score(X, labels)):.3f}"))

    '''STAGE 3: PLOTTING MY CLUSTERS ON A MAP USING FOLIUM.'''
    location = acc_glasgow_year_dbscan["Latitude"].mean(), acc_glasgow_year_dbscan["Longitude"].mean()
    cluster_map = folium.Map(location=location, zoom_start=13)
    folium.TileLayer("cartodbpositron").add_to(cluster_map)
    clust_colours = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
                     '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#7a7a6f', '#b15928']
    for i in range(0,
                   len(acc_glasgow_year_dbscan)):  # this basically creates the clusters on the map, ignoring the -1 labeled accidents
        colouridx = acc_glasgow_year_dbscan["Cluster"].iloc[i]
        if colouridx == -1:  # basically just skipping the unclustered accidents ("noise")
            pass
        else:  # printing onto the map
            col = clust_colours[colouridx % len(clust_colours)]
            folium.CircleMarker([acc_glasgow_year_dbscan["Latitude"].iloc[i],
                                 acc_glasgow_year_dbscan["Longitude"].iloc[i]],
                                radius=10, color=col, fill=col).add_to(cluster_map)

    output_name = str(
        f"Glasgow_all_years_samples_{b}_distance_{c}.html")  # f-strings has got to be the best thing i've learnt in this entire project
    cluster_map.save(output_name)


# This is my function to plot accidents on maps for unsupervised learning using DBSCAN
# a = DF in question
# b = min_samples (default = 5), increase to lower clusters
# c = distance between accidents, decrease to lower clusters
func_unsupervised_dbscan_all_years(glasgow, 10, 300)

toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))

'''SUPERVISED LEARNING'''
'''3.0 LogReg'''


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


'''3.1 AdaBoost'''


#LOADING DATA AND CLEANING IT/SKIMMING IT DOWN (HAD TO DROP ALL USELESS COLUMNS (SOME HAD NA AS WELL)


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
# ADABOOST "Boosting-based Ensemble learning: sequential learning technique"
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

'''3.2 XGBoost'''


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


# Creating split for training and testing
x_train, x_test, y_train, y_test = train_test_split(glasgow_x, glasgow_y, test_size=0.3)
D_train = xgb.DMatrix(x_train, label=y_train)
D_test = xgb.DMatrix(x_test, label=y_test)




#ASCERTAINING WHICH CLASSIFIER TO USE:'''
#model = XGBClassifier()
#model.fit(x_train, y_train)
#print(model.objective)

#MULTI:SOFTPROB'''

param = {
    'eta': 0.1,
    'max_depth': 3,
    'objective': 'multi:softprob',
    'num_class': 4}

steps = 20

#Running XGBoost: Notes are at the borrom for params'''

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