import timeit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
