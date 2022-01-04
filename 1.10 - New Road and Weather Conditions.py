import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
