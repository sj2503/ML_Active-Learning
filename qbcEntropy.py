import pandas as pd
import csv
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import layers
from keras.utils import np_utils
import sys


def prepare_modified_data(filename1,filename2):

    
    df_ = pd.read_csv(filename1,header=None)
    list_ = []

    for i in df_.iloc[:].values:
        list_.append(int(i))
    
    data_modified = pd.read_csv(filename2,index_col=0)
    data_modified = data_modified.iloc[list_,:]
    data = np.array(data_modified)
    m,n = data.shape
    number_rows = int(0.1*m)
    data_dev = data[:number_rows]

    Y_dev = data_dev[:,0]
    X_dev = data_dev[:,1:785]

    data_test = data[number_rows:]

    Y_test = data_test[:,0]
    X_test = data_test[:,1:785]
    
    return X_test,Y_test,X_dev,Y_dev,data_test

def prepare_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    m,n = data.shape
    np.random.shuffle(data)
    number_rows = int(0.1*m)
    data_dev = data[:number_rows]

    Y_dev = data_dev[:,0]
    X_dev = data_dev[:,1:785]

    data.to_csv("QBC_initial_file.csv")

    data_test = data[number_rows:]

    Y_test = data_test[:,0]
    X_test = data_test[:,1:785]
    
    return X_test,Y_test,X_dev,Y_dev,data_test

def prepare_data_again(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    m,n = data.shape
    number_rows = int(0.1*m)
    data_dev = data[:number_rows]

    Y_dev = data_dev[:,0]
    X_dev = data_dev[:,1:785]

    data_test = data[number_rows:]

    Y_test = data_test[:,0]
    X_test = data_test[:,1:785]
    
    return X_test,Y_test,X_dev,Y_dev,data_test 


def vote(arr):
    for i in range(arr.shape[0]):
        tempo = arr[i]
        ind = np.argmax(tempo)
        arr[i][ind] = 1
        arr[i][arr[i]<1] = 0
    
    return arr



def train_first_time():
    #Open the csv file and read it into a pandas dataframe
    df = pd.read_csv("data.csv")

    # Select 10% of the rows randomly
    train_data = df.sample(frac=0.1)

    # Remove the selected rows from the original DataFrame
    remaining_data = df.drop(train_data.index)

    # Save the remaining rows to a CSV file
    remaining_data.to_csv("Remaining_Data_Keras.csv")

    # Define the input and output columns of the data
    input_cols = ["pixel" + str(i) for i in range(1, 785)]
    output_cols = ["label"]

    # Get the input and output data for training and testing
    train_input = train_data[input_cols]
    train_output = train_data[output_cols]
    test_input = remaining_data[input_cols]
    test_output = remaining_data[output_cols]

    # Define the deep neural network model
    model1 = Sequential()
    model1.add(Dense(64, input_dim=len(input_cols), activation="relu"))
    model1.add(Dense(32, activation="relu"))
    model1.add(Dense(32, activation="relu"))
    model1.add(Dense(16, activation="relu"))
    model1.add(Dense(10, activation="softmax"))

    # Compile the model
    model1.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model1.fit(train_input, train_output, epochs=50, batch_size=128)

    # Evaluate the model on the test data
    model1.evaluate(test_input, test_output)

    model1.save("model21.h5")

    # Define the deep neural network model
    model2 = Sequential()
    model2.add(Dense(64, input_dim=len(input_cols), activation="relu"))
    model2.add(Dense(32, activation="relu"))
    model2.add(Dense(32, activation="sigmoid"))
    model2.add(Dense(10, activation="softmax"))

    # Compile the model
    model2.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    # Train the model
    model2.fit(train_input, train_output, epochs=50, batch_size=128)

    # Evaluate the model on the test data
    model2.evaluate(test_input, test_output)

    model2.save("model22.h5")

    # Define the deep neural network model
    model3 = Sequential()
    model3.add(Dense(64, input_dim=len(input_cols), activation="relu"))
    model3.add(Dense(64, activation="relu"))
    model3.add(Dense(32, activation="relu"))
    model3.add(Dense(10, activation="softmax"))

    # Compile the model
    model3.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model3.fit(train_input, train_output, epochs=50, batch_size=64)

    # Evaluate the model on the test data
    model3.evaluate(test_input, test_output)

    model3.save("model23.h5")

    # Define the deep neural network model
    model4 = Sequential()
    model4.add(Dense(64, input_dim=len(input_cols), activation="relu"))
    model4.add(Dense(64, activation="relu"))
    model4.add(Dense(32, activation="relu"))
    model4.add(Dense(16, activation="sigmoid"))
    model4.add(Dense(10, activation="softmax"))

    # Compile the model
    model4.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    # Train the model
    model4.fit(train_input, train_output, epochs=50, batch_size=128)

    # Evaluate the model on the test data
    model4.evaluate(test_input, test_output)

    model4.save("model24.h5")

    # Define the deep neural network model
    model5 = Sequential()
    model5.add(Dense(64, input_dim=len(input_cols), activation="relu"))
    model5.add(Dense(32, activation="relu"))
    model5.add(Dense(16, activation="relu"))
    model5.add(Dense(16, activation="sigmoid"))
    model5.add(Dense(10, activation="softmax"))

    # Compile the model
    model5.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model5.fit(train_input, train_output, epochs=50, batch_size=64)

    # Evaluate the model on the test data
    model5.evaluate(test_input, test_output)

    model5.save("model25.h5")

    predict1 = model1.predict(train_input)
    predict2 = model2.predict(train_input)
    predict3 = model3.predict(train_input)
    predict4 = model4.predict(train_input)
    predict5 = model5.predict(train_input)

    df1 = pd.DataFrame(predict1)
    df2 = pd.DataFrame(predict2)
    df3 = pd.DataFrame(predict3)
    df4 = pd.DataFrame(predict4)
    df5 = pd.DataFrame(predict5)

    overall_logs = pd.concat([df1,df2,df3,df4,df5], axis=1)

    overall_logs.to_csv("overall_logs.csv", index=True, header=True)

    df1["max_index1"] = df1.apply(lambda x: x.argmax(), axis=1)
    df2["max_index2"] = df2.apply(lambda x: x.argmax(), axis=1)
    df3["max_index3"] = df3.apply(lambda x: x.argmax(), axis=1)
    df4["max_index4"] = df4.apply(lambda x: x.argmax(), axis=1)
    df5["max_index5"] = df5.apply(lambda x: x.argmax(), axis=1)

    df_concat = pd.concat([df1[["max_index1"]], df2[["max_index2"]], df3[["max_index3"]], df4[["max_index4"]], df5[["max_index5"]]], axis=1)

    df_vote_count = pd.DataFrame(columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    df_vote_count["0"] = df_concat.apply(lambda x: (x==0).sum(), axis=1)
    df_vote_count["1"] = df_concat.apply(lambda x: (x==1).sum(), axis=1)
    df_vote_count["2"] = df_concat.apply(lambda x: (x==2).sum(), axis=1)
    df_vote_count["3"] = df_concat.apply(lambda x: (x==3).sum(), axis=1)
    df_vote_count["4"] = df_concat.apply(lambda x: (x==4).sum(), axis=1)
    df_vote_count["5"] = df_concat.apply(lambda x: (x==5).sum(), axis=1)
    df_vote_count["6"] = df_concat.apply(lambda x: (x==6).sum(), axis=1)
    df_vote_count["7"] = df_concat.apply(lambda x: (x==7).sum(), axis=1)
    df_vote_count["8"] = df_concat.apply(lambda x: (x==8).sum(), axis=1)
    df_vote_count["9"] = df_concat.apply(lambda x: (x==9).sum(), axis=1)

    df_vote_count.to_csv("overall_votes.csv", index=True, header=True)


def train_second_time(filename1):
    model1 = keras.models.load_model("model21.h5")
    model2 = keras.models.load_model("model22.h5")
    model3 = keras.models.load_model("model23.h5")
    model4 = keras.models.load_model("model24.h5")
    model5 = keras.models.load_model("model25.h5")

    X_test,Y_test,X_dev,Y_dev,data_rem = prepare_modified_data(filename1, "Remaining_Data_Keras.csv")


    # encoder = LabelEncoder()
    # encoder.fit(Y_dev)
    # encoded_Y = encoder.transform(Y_dev)
    # Y_Dev = np_utils.to_categorical(encoded_Y)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_dev = min_max_scaler.fit_transform(X_dev)

    # X_test = min_max_scaler.fit_transform(X_test)    



    history1 = model1.fit(X_dev,Y_dev,epochs=50,batch_size=1,verbose=0)
    history2 = model2.fit(X_dev,Y_dev,epochs=50,batch_size=1,verbose=0)
    history3 = model3.fit(X_dev,Y_dev,epochs=50,batch_size=1,verbose=0) 
    history4 = model4.fit(X_dev,Y_dev,epochs=50,batch_size=1,verbose=0)
    history5 = model5.fit(X_dev,Y_dev,epochs=50,batch_size=1,verbose=0)
    
    arr1 = model1.predict(X_test)
    arr2 = model2.predict(X_test)
    arr3 = model3.predict(X_test)
    arr4 = model4.predict(X_test)
    arr5 = model5.predict(X_test)

    file1 = pd.DataFrame(arr1)
    file2 = pd.DataFrame(arr2)
    file3 = pd.DataFrame(arr3)
    file4 = pd.DataFrame(arr4)
    file5 = pd.DataFrame(arr5)

    overall_logs = pd.concat([file1,file2,file3,file4,file5],axis=1)

    arr1 = vote(arr1)
    arr2 = vote(arr2)
    arr3 = vote(arr3)
    arr4 = vote(arr4)
    arr5 = vote(arr5)

    overall_votes = arr1 + arr2 + arr3 + arr4 + arr5
    overall_votes = pd.DataFrame(overall_votes)

    model1.save("model21.h5")
    model2.save("model22.h5")
    model3.save("model23.h5")
    model4.save("model24.h5")
    model5.save("model25.h5")

    overall_logs.to_csv("overall_logs.csv")
    overall_votes.to_csv("overall_votes.csv")

    df_remaining_data = pd.DataFrame(data_rem)
    df_remaining_data.to_csv("Remaining_Data_Keras.csv")

    



def main(argv1,argv2):
    if(argv1=="new"):
        train_first_time()
    elif(argv1=="retrain"):
        train_second_time(argv2)



if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])