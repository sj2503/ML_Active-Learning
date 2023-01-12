import pandas as pd
import csv
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def main():
    # Open the csv file and read it into a pandas dataframe
    df = pd.read_csv("ninety_uld.csv",index_col=0)

    df = df.sample(frac=1)

    df_fourty = df.sample(frac=0.4)

    # Define the input and output columns of the data
    input_cols = ["pixel" + str(i) for i in range(1, 785)]
    output_cols = ["label"]

    init_model = keras.models.load_model("model1.h5")

    print(df_fourty)
    print(df_fourty.shape)
    df_fourtyunlabelled = df_fourty[input_cols]
    df_fourtylabelled = df_fourty[output_cols]

    # Create a k-NN clustering model with 5 clusters and 10 iterations
    kmeans = KMeans(n_clusters=10, n_init=10)

    # Train the clustering model using the data
    kmeans.fit(df_fourtyunlabelled)

    # Use the model to generate cluster labels for the data
    cluster_labels = kmeans.predict(df_fourtyunlabelled)

    df_fourtyunlabelled["Cluster_Label"] = cluster_labels

    df_fourtyunlabelled["Prediction"] = np.full(len(df_fourtyunlabelled), 0)

    for idx in range(0,10):
        df_temp = df_fourtyunlabelled[df_fourtyunlabelled["Cluster_Label"] == idx]
        df_twenty = df_temp.iloc[:(int)(0.2 * len(df_temp)), :]
        pred = init_model.predict(df_twenty.iloc[:, :-2])
        # print(df_twenty.shape)
        # print(pred.shape)
        # print(df_twenty)
        # print(pred)

        predi = []
        for ten_prediction in pred:
            max_index = np.argmax(ten_prediction)
            predi.append(max_index)
        
        df_twenty = df_twenty.assign(Prediction = predi)
        counts = df_twenty["Prediction"].value_counts()
        max_count = counts.idxmax()
        y = np.full((len(df_temp),), max_count)
        df_temp["Prediction"] = max_count
        df_fourtyunlabelled.loc[df_fourtyunlabelled["Cluster_Label"].isin([idx]),"Prediction"] = max_count

    df_fourtylabelled["Prediction"] = df_fourtyunlabelled["Prediction"]

    df_fourtylabelled.to_csv("k_cluster.csv")


if __name__ == "__main__":
    main()