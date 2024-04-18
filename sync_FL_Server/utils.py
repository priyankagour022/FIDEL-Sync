import os
import grpc
import functions_pb2
import functions_pb2_grpc
import base64
import scipy.io as sio
import concurrent.futures


channels = []
stubs = []
iteration = 0
n = 0

"""Util functions"""

def init(ipPortOfFIDEL_client):
    for x in ipPortOfFIDEL_client:
        channels.append(grpc.insecure_channel(x))

    for i in range(0, len(ipPortOfFIDEL_client)):
        stubs.append(functions_pb2_grpc.FederatedAppStub(channels[i]))

    global n
    n = len(stubs)
    

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



def initClientFun(i):
    """ This method is not beign used"""
    client_param = functions_pb2.Number(value = i)
    res = stubs[i].InitilizeClients(client_param)
    print("Device {} is initilized with status {}".format(res.id, res.status))

 
def trainFunc(i):
    print("in utils.trainFunc")
    filename = "model.h5"
    train_parm = functions_pb2.TrainTriplet(id = i, time = iteration, model = encode_file(filename))
    print("after train_arm")
    res = stubs[i].Train(train_parm)

    print("opening file")
    with open("Models/model_"+str(i)+".h5","wb") as file:
        print("opened successfully")
        file.write(base64.b64decode(res.model))
    print("Training result of device {} for {} th iteration".format(res.id, res.time)) 


def getDataFunc(i):
    """ This method is not beign used"""
    client_param = functions_pb2.Number(value = i)
    res = stubs[i].FetchClientResults(client_param)
    print("Data comming from client : ", i, res.model)


# def getTestDataset():

#     # print("Inside testdata")

#     database = sio.loadmat('../IEEE for federated learning/data_base_all_sequences_random.mat')
    
#     X = database['Data_test_2']
#     y = database['label_test_2']

#     return X, 

def getTestDataset():

    # print("Inside testdata")

    database = sio.loadmat('../IEEE for federated learning/data_base_all_sequences_random.mat')
    
    X = database['Data_test_2']
    y = database['label_test_2']

    return X, y




#Utility function to convert model(h5) into string
def encode_file(file_name):
    with open('Models/'+file_name,'rb') as file:
        encoded_string = base64.b64encode(file.read())
    return encoded_string