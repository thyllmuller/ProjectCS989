import numpy as np
import pandas as pd
import folium
from geopy.distance import great_circle
from sklearn import metrics
from sklearn.cluster import DBSCAN as dbscan
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import timeit

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
