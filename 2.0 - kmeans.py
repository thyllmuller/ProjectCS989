import timeit
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

''' THE ELBOW CURVE'''
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

''' THE KMEANS '''
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
