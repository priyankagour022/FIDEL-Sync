import os
import utils as utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import json
from time import sleep, time
from PIL import Image
import tensorflow as tf  
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv1D, Dropout, Reshape, MaxPooling1D, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import cv2



def saveLearntMetrices(modelName):
     
    # print("Inside save paramertes")
    model = load_model(modelName)
    X_test, y_test = utils.getTestDataset()
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



# Function for preprocessing the human detection dataset for each client
def preprocess_data_for_client(client_id, data_dir, img_height=64, img_width=64):
    images = []
    labels = []

    # Construct the path for the client's data directory
    client_data_dir = os.path.join(data_dir, str(client_id))

    # Iterate through each image in the client's data directory
    for image_name in os.listdir(client_data_dir):
        image_path = os.path.join(client_data_dir, image_name)
        # Load and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (img_height, img_width))
        image = image.astype(np.float32) / 255.0  # Normalize pixel values
        images.append(image)
        labels.append(client_id)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# def createData():

#     #Data preprocessing
#     database = sio.loadmat('../IEEE for federated learning/data_base_all_sequences_random.mat')
#     x_train = database['Data_train_2']
#     y_train = database['label_train_2']

#     indices = database['permut']
#     indices = indices - 1 # 1 is subtracted to make 0 at index
#     indices = indices[0] # Open indexing

#     i = 0
#     slot = len(indices)//n
#     data = []
#     folderId = 0
#     if not os.path.exists('../data'):
#         os.makedirs('../data')
#     while i < len(indices) :
#         if i//slot + 1 <= n:
#             folderId = i//slot + 1
#             data = []
#         else:
#             folderId = n

#         for _ in range(slot):
#             x = x_train[indices[i]]
#             y = y_train[indices[i]]
#             row = np.append(x, y)
#             data.append(row)
#             i = i + 1
#             if(i == len(indices)):
#                 break
#         if not os.path.exists('../data/node0' + str(folderId)):
#             os.makedirs('../data/node0' + str(folderId))
#         df = pd.DataFrame(data)
#         df.reset_index()
#         df.to_csv('../data/node0' + str(folderId) + '/data.csv')
#     print("Dataset is created for %d devices" %(n))
  



# This fucntion aggregates all models parmeters and create new optimized model 
def fedAvg():
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

    K.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(512,)))
    model.add(tf.keras.layers.Dense(8))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.Dense(8, activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.save('Models/model.h5')

    model.summary()



def initlizeGlobalMetrics():
    metric = {'accuracy' : [], 'loss' : []}
    
    with open('Models/globalMetrics.txt', "w") as f:
        f.write(json.dumps(metric))

def visualizeTraining():
    # path = "/home/aditya/Desktop/DN_Sync_result_1/"
    print("Inside visualize")    
    # fp =  open(path + 'server/globalMetrics.txt','r')
    fp =  open('Models/globalMetrics.txt','r')

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
    for i in range(1, n + 1):
        with open("../data/node0" + str(i) + '/metrics.txt', 'r') as f:
            trainMetrics = json.load(f)
            plt.plot(trainMetrics['accuracy'], label='Device ' + str(i) )
    plt.plot(gloablMetrics['accuracy'], '--b', label='Server')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Rounds')
    plt.legend()
    h.show()

    s = plt.figure(4)
    for i in range(1, n + 1):
        with open("../data/node0" + str(i) + '/metrics.txt', 'r') as f:
            trainMetrics = json.load(f)
            plt.plot(trainMetrics['loss'], label='Device ' + str(i) )
    plt.plot(gloablMetrics['loss'], '--b', label='Server')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Rounds')
    plt.legend()
    s.show()




#declare utils.channels here
utils.init([
    'localhost:8081',
    'localhost:8082',
    'localhost:8083',
    'localhost:8084',
    'localhost:8085'
])

n = len(utils.stubs)

if __name__ == '__main__':
    
# User options for training main()
    while True:

        print("1. Initiate initialization")
        print("2. Perform training on all nodes")
        print("3. Average and optimize new model")
        print("4. Visualize model accuracy/loss")
        print("5. Batch training")
        print("6. Exit")
        print("0. Fetech data from clients")
        print("Enter an option: ")
        option = input()

        if (option == "1"):
            utils.initilizeClients()
            createInitialModel()
            initlizeGlobalMetrics()
            saveLearntMetrices('Models/model.h5')
            # createData() # To create partationed of dataset for clients
        if (option == "2"):
            utils.train()
        if (option == "3"):
            fedAvg()
        if (option == "4"):
            visualizeTraining()
        if (option == "5"):
            for i in range(60):
                print("Current Round ", i+1)
                utils.train()
                fedAvg()
        if (option == "6"):
            break
        if (option == "0"):
            print("Function is not implemented")
            # getDataFromClients()
                