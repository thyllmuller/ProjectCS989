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
