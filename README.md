# FEDIL-Sync - A Federated learning framework for neural network training

The framework is designed to train a neural network task on various fog nodes. The fog nodes consist of resources constrained devices such as Raspberry Pi.

The standard federated learning paradigm with fedAvg implementation is done on continuously generating datasets. 

Firstly, the server starts training by initializing the model and asking connected nodes to train the model on their local datasets. The server waits for participating clients to return the model, thereafter it aggregates the updates. The training continues till the global model converges or the expected model is created. 

## Structure and pre-requisite

The project contains code for the client and server that need to be run on respective devices. The clients are resource-constrained devices such as Raspberry Pi, and a server can be computers/laptops.  

All participating nodes should have docker installed so that appropriate images can be downloaded. Otherwise, the user has to create a docker image with the code given in the project. 

To connect to the client, the server need to have access to ip address of the participating clients. 

Additionally, every client should have data in the data folder that will be used for training. 

# How to run the code

## On the server side
1. Run requiremetn.txt
2. Edit/create appropriate channels for available clients with respective ip addresses in server.py
3. Define neural network in the createInitialModel() function.
4. Run server.py


## On the client side 

### To run directly on the client 
1. Check/change dockr image in docker-compose.yaml file
2. Run docker-compose.yaml file

The compose file will be downloaded and executed on the fog device. 


### To create a new image

In case of a given image is not working, then the image can be created with docker-compose.yaml file. 

Edit compose file on the image tag with build code. 
```
build:
    context: .
    dockerfile: Dockerfile
```


The program is still menu driven. To start the execution, you need to hit start from the server end. In the current form, press 7, which will execute n(60) rounds of federated learning. 


## Result  and report

The server will have evaluation results(accuracy, loss) and latest global model in Models folder.  

Similarly, every client will have locally trained results and latest model in the data folder. 





