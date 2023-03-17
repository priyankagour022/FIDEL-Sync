import os
import grpc
import base64

import functions_pb2
import functions_pb2_grpc
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import json
from time import sleep, time


import warnings  
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf  
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dense, Conv1D, Dropout, Reshape, MaxPooling1D, Flatten
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import to_categorical


#Utility function to convert model(h5) into string
def encode_file(file_name):
    with open('Models/'+file_name,'rb') as file:
        encoded_string = base64.b64encode(file.read())
    return encoded_string


#declare channels here
#channel<n> is the channel for node 'n'
channel01 = grpc.insecure_channel('192.168.0.204:8080')
channel02 = grpc.insecure_channel('192.168.0.192:8080')
channel03 = grpc.insecure_channel('192.168.0.106:8080')
channel04 = grpc.insecure_channel('192.168.0.251:8080')
channel05 = grpc.insecure_channel('192.168.0.23:8080')

#declare stubs here
#stub<n> is the stub for channel<n>
stub01 = functions_pb2_grpc.FederatedAppStub(channel01)
stub02 = functions_pb2_grpc.FederatedAppStub(channel02)
stub03 = functions_pb2_grpc.FederatedAppStub(channel03)
stub04 = functions_pb2_grpc.FederatedAppStub(channel04)
stub05 = functions_pb2_grpc.FederatedAppStub(channel05)

# array of all our stubs
stubs = [
    stub01,
    stub02,
    stub03,
    stub04,
    stub05
]

#number of nodes on the network
n = len(stubs)
# n = 5
iteration = 0

"""Util functions"""

def initClientFun(i):
    """ This method is not beign used"""
    client_param = functions_pb2.Number(value = i)
    res = stubs[i].InitilizeClients(client_param)
    print("Device {} is initilized with status {}".format(res.id, res.status))

def trainFunc(i):
    filename = "model.h5"
    train_parm = functions_pb2.TrainTriplet(id = i, time = iteration, model = encode_file(filename))
    res = stubs[i].Train(train_parm)

    with open("Models/model_"+str(i)+".h5","wb") as file:
        file.write(base64.b64decode(res.model))
    print("Training result of device {} for {} th iteration".format(res.id, res.time))  

def getDataFunc(i):
    """ This method is not beign used"""
    client_param = functions_pb2.Number(value = i)
    res = stubs[i].FetchClientResults(client_param)
    print("Data comming from client : ", i, res.model)

def getTestDataset():

    # print("Inside testdata")

    database = sio.loadmat('../IEEE for federated learning/data_base_all_sequences_random.mat')
    
    X = database['Data_test_2']
    y = database['label_test_2']

    return X, y


def saveLearntMetrices(modelName):
     
    # print("Inside save paramertes")
    model = load_model(modelName)
    X_test, y_test = getTestDataset()
    y_test = to_categorical(y_test)
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Agreegated model with all data loss : {} and accuracy : {}".format(score[0], score[1]))

    with open('Models/globalMetrics.txt','r+') as f:
        trainMetrics = json.load(f)
        trainMetrics['accuracy'].append(score[1])
        trainMetrics['loss'].append(score[0])
        f.seek(0) 
        f.truncate()
        f.write(json.dumps(trainMetrics))


def createData():

    #Data preprocessing
    database = sio.loadmat('../IEEE for federated learning/data_base_all_sequences_random.mat')
    x_train = database['Data_train_2']
    y_train = database['label_train_2']
    #y_train_t = to_categorical(y_train)
    #x_train = (x_train.astype('float32') + 140) / 140 # DATA PREPARATION (NORMALIZATION AND SCALING OF FFT MEASUREMENTS)
    #x_train2 = x_train[iii * samples:((iii + 1) * samples - 1), :] # DATA PARTITION

    # x_test = database['Data_test_2']
    # y_test = database['label_test_2']
    #x_test = (x_test.astype('float32') + 140) / 140
    #y_test_t = to_categorical(y_test)

    indices = database['permut']
    indices = indices - 1 # 1 is subtracted to make 0 at index
    indices = indices[0] # Open indexing

    i = 0
    slot = len(indices)//n
    data = []
    folderId = 0
    if not os.path.exists('../data'):
        os.makedirs('../data')
    while i < len(indices) :
        if i//slot + 1 <= n:
            folderId = i//slot + 1
            data = []
        else:
            folderId = n

        for _ in range(slot):
            x = x_train[indices[i]]
            y = y_train[indices[i]]
            row = np.append(x, y)
            data.append(row)
            i = i + 1
            if(i == len(indices)):
                break
        if not os.path.exists('../data/node0' + str(folderId)):
            os.makedirs('../data/node0' + str(folderId))
        df = pd.DataFrame(data)
        df.reset_index()
        df.to_csv('../data/node0' + str(folderId) + '/data.csv')
    print("Dataset is created for %d devices" %(n))
  



""" These functions are called based on user input """

# Create local data for every participating device
def initilizeClients():

    executor = concurrent.futures.ProcessPoolExecutor(n)
    futures = [executor.submit(initClientFun, i) for i in range(n)]
    concurrent.futures.wait(futures)

# Send latest model to all participating device 
def sendModel(opt):

    executor = concurrent.futures.ProcessPoolExecutor(n)
    futures = [executor.submit(sendFunc, i, opt) for i in range(n)]
    concurrent.futures.wait(futures)
    
# Call for training for all participating devices
def train():
    try :
        executor = concurrent.futures.ProcessPoolExecutor(n)
        futures = [executor.submit(trainFunc, i) for i in range(n)]
        concurrent.futures.wait(futures, return_when='ALL_COMPLETED')
        print("Out of training")
    except Exception as e:
        print(e)
        print("An error occured!")
        return 1

def getDataFromClients():
    executor = concurrent.futures.ProcessPoolExecutor(n)
    futures = [executor.submit(getDataFunc, i) for i in range(n)]
    concurrent.futures.wait(futures, return_when="ALL_COMPLETED")


# This fucntion aggregates all models parmeters and create new optimized model 
def optimiseModels():
    models = list()

    models = [load_model("Models/model_"+str(i)+".h5") for i in range(n)]
    [os.remove("Models/model_"+str(i)+".h5") for i in range(n)] 
    weights = [model.get_weights() for model in models]

    new_weights = list()
    print("Total aggregation clients", len(weights))
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))

    new_model = models[0]  
    new_model.set_weights(new_weights)
    new_model.save("Models/model.h5")

    # new_model.save("Models/optimised_model.h5")
    print("Averaged over all models - optimised model saved!")
    saveLearntMetrices("Models/model.h5")

   
#Create and initilize model for first time. 
def createInitialModel():

    # K.clear_session()
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(512,)))
    # model.add(tf.keras.layers.Dense(8, activation='softmax'))
    # model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # model.save('Models/model.h5')

    K.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(512,)))
    model.add(tf.keras.layers.Dense(8))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.Dense(8, activation='softmax'))
    model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.save('Models/model.h5')

    # K.clear_session()
    # model = Sequential()
    # model.add(Reshape((512,1), input_shape=(512,1)))
    # model.add(Conv1D(filters=8, kernel_size=3,padding='same', activation='relu', input_shape=(512,1)))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=4, kernel_size=3, padding='same',  activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(8, activation='softmax'))
    # model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # model.save('Models/model.h5')


    model.summary()




def initlizeGlobalMetrics():
    metric = {'accuracy' : [], 'loss' : []}
    
    with open('Models/globalMetrics.txt', "w") as f:
        f.write(json.dumps(metric))

def visualizeTraining():
    path = "/home/aditya/Desktop/DN_Sync_result_1/"
    print("Inside visualize")    
    fp =  open(path + 'server/globalMetrics.txt','r')
    gloablMetrics = json.load(fp)

    f = plt.figure(1)
    plt.plot(gloablMetrics['accuracy'], label='Test')
    plt.title('Global model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Rounds')
    plt.legend()
    f.show()
    
    g = plt.figure(2)
    plt.plot(gloablMetrics['loss'], label='Test')
    plt.title('Global loss')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')
    plt.legend()
    g.show()


    h = plt.figure(3)
    for i in range(0, n):
        with open(path + "client" + str(i) + '/metrics.txt', 'r') as f:
            trainMetrics = json.load(f)
            plt.plot(trainMetrics['accuracy'], label='Device ' + str(i) )
    plt.plot(gloablMetrics['accuracy'], '--b', label='Server')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Rounds')
    plt.legend()
    h.show()

    s = plt.figure(4)
    for i in range(0, n):
        with open(path + "client" + str(i) + '/metrics.txt', 'r') as f:
            trainMetrics = json.load(f)
            plt.plot(trainMetrics['loss'], label='Device ' + str(i) )
    plt.plot(gloablMetrics['loss'], '--b', label='Server')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')
    plt.legend()
    s.show()


if __name__ == '__main__':
    
# User options for training main()
    while True:

        print("1. Initiate initialization")
        print("2. Initialize model on all nodes")
        print("3. Perform training on all nodes")
        print("4. Average and optimize new model")
        print("5. Send new model to all nodes")
        print("6. Visualize model accuracy/loss")
        print("7. Batch training")
        print("8. Exit")
        print("0. Fetech data from clients")
        print("Enter an option: ")
        option = input()

        if (option == "1"):
            initilizeClients()
            createInitialModel()
            initlizeGlobalMetrics()
            saveLearntMetrices('Models/model.h5')
            # createData() # To create partationed of dataset for clients
        if (option == "2"):
            # createData() # To create partationed of dataset for clients
            print("This option is obsolute")
            # sendModel(int(option))
        if (option == "3"):
            train()
        if (option == "4"):
            optimiseModels()
        if (option == "5"):
            print("This option is obsolute")
            # sendModel(int(option))
        if (option == "6"):
            visualizeTraining()
        if (option == "7"):
            for i in range(60):
                print("Current Round ", i+1)
                train()
                optimiseModels()
        if (option == "8"):
            break
        if (option == "0"):
            print("Function is not implemented")
            # getDataFromClients()
                