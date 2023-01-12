import pandas as pd
import csv
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def build_data_repeat(filename):
    data = pd.read_csv(filename,index_col=0)
    data = np.array(data)
    m,n = data.shape

    no_rows = int(0.1*m)
    data_dev = data[:no_rows].T

    dev_Y = data_dev[0]
    dev_X = data_dev[1:785]

    test_data = data[no_rows:].T

    test_Y = test_data[0]
    test_X = test_data[1:785]

    
    return test_X,test_Y,dev_X,dev_Y,test_data.T


def prepare_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    m,n = data.shape
    np.random.shuffle(data)
    no_rows = int(0.1*m)
    data_dev = data[:no_rows].T

    dev_Y = data_dev[0]
    dev_X = data_dev[1:785]

    test_data = data[no_rows:].T

    test_Y = test_data[0]
    test_X = test_data[1:785]

    df_initial = pd.DataFrame(data)
    df_initial.to_csv("samefile.csv")
    
    return test_X,test_Y,dev_X,dev_Y,test_data.T


def prepare_modified_data(file1,file2):
    df_ = pd.read_csv(file1,header=None)
    list_ = []

    for i in df_.iloc[:].values:
        list_.append(int(i))
    
    data_mod = pd.read_csv(file2,index_col=0)
    data_mod = data_mod.iloc[list_,:]

    data = np.array(data_mod)
    m,n = data.shape
    no_rows = int(0.1*m)
    data_dev = data[:no_rows].T

    dev_Y = data_dev[0]
    dev_X = data_dev[1:785]

    test_data = data[no_rows:].T

    test_Y = test_data[0]
    test_X = test_data[1:785]
    
    return test_X,test_Y,dev_X,dev_Y,test_data.T



def init_func():
  # Open the csv file and read it into a pandas dataframe
  df = pd.read_csv("data.csv")

  # Select 10% of the rows randomly
  train_data = df.sample(frac=0.1)

  # Remove the selected rows from the original DataFrame
  rest_data = df.drop(train_data.index)

  # Save the remaining rows to a CSV file
  rest_data.to_csv("fashion-mnist_ninety.csv")

  # Define the input and output columns of the data
  input_cols = ["pixel" + str(i) for i in range(1, 785)]
  output_cols = ["label"]

  train_input = train_data[input_cols]
  train_output = train_data[output_cols]
  test_input = rest_data[input_cols]
  test_output = rest_data[output_cols]

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

  model1.save("model1.h5")

  predict = model1.predict(train_input)

  df.to_csv("predic.csv", index=True, header=False)

  return model1


def accuracy(model1, test_Y,test_X):
    correct = 0
    test_X = test_X.T
    prediction = model1.predict(test_X)

    for i in range(test_X.shape[0]):
        if(prediction[i].argmax() == test_Y[i]): 
            correct += 1

    return correct,test_X.shape[0]

def load_wgts(filename):
    saved_weights = np.load(filename,allow_pickle=True)
    W1 = saved_weights[0]
    b1 = saved_weights[1]
    W2 = saved_weights[2]
    b2 = saved_weights[3]
    W3 = saved_weights[4]
    b3 = saved_weights[5]
    W4 = saved_weights[6]
    b4 = saved_weights[7]
    W5 = saved_weights[8]
    b5 = saved_weights[9]

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5


def init_parameters():
    W1 = np.random.randn(64,784)
    b1 = np.random.randn(64,1)
    W2 = np.random.randn(32,64)
    b2 = np.random.randn(32,1)
    W3 = np.random.randn(32,64)
    b3 = np.random.randn(32,1)
    W4 = np.random.randn(16,32)
    b4 = np.random.randn(16,1)
    W5 = np.random.randn(10,16)
    b5 = np.random.randn(10,1)

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5



def main(argument,csvf):

    #training
    alpha = 0.2
    num_iters = 10
    loss = 'mse' # 'cross_entropy' |  'mse'


    # min_max_scaler = preprocessing.MinMaxScaler()
    # utility functions


    if argument=="new" and csvf=="nofile":
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5  = init_parameters()
        weights_list = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5]
        weights_list = np.array(weights_list,dtype=object)
        np.save("liw_initial.npy",weights_list,allow_pickle=True)
        model1 = init_func()
        np.save("liw_initial.npy",weights_list,allow_pickle=True)    
    elif argument=="retrain":
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = load_wgts("liw_same.npy") 
    elif argument=="new" and csvf=="samefile.csv":
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = load_wgts("liw_initial.npy") 

        

    if argument=="new" and csvf=="nofile":
        test_X,test_Y,dev_X,dev_Y,data_rem = prepare_data("data.csv")
    elif argument=="retrain":
        test_X,test_Y,dev_X,dev_Y,data_rem = prepare_modified_data(csvf,"Remaining_data.csv")
    elif argument=="new" and csvf=="samefile.csv":
        test_X,test_Y,dev_X,dev_Y,data_rem = build_data_repeat(csvf)

    list_weights = []

    acc1,total1 = accuracy(model1,dev_Y,dev_X)
    acc,total = accuracy(model1,test_Y, test_X)

    f = open("Learning_Curves.txt", "a")
    f.write(f"Train Accuracy:{acc1}/{total1} {csvf}\n")
    f.write(f"Test Accuracy : {acc}/{total} {csvf}\n\n")

    f.close()


    df = pd.DataFrame(list_weights)
    df_rest_data = pd.DataFrame(data_rem)
    # create a list of column names
    col_names = []
    col_names.append("label")
    for i in range(1,785):
        col_names.append("pixel" + str(i))
    # rename the columns in the dataframe
    df_rest_data.columns = col_names

    if argument=="new":
        df_rest_data.to_csv("ninety_uld.csv")

    df_rest_data.to_csv("Remaining_Data.csv")
    df.to_csv("predic.csv")
    weights_list = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5]
    weights_list = np.array(weights_list,dtype=object)
    


    if argument=="new":
        np.save("liw_.npy",weights_list,allow_pickle=True)    
        
        np.save("liw_same.npy",weights_list,allow_pickle= True)
    elif argument=="retrain":
        np.save("liw_same.npy",weights_list,allow_pickle=True)
    



if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])

