from urllib import response
import grpc
from concurrent import futures
import time

import functions_pb2
import functions_pb2_grpc
import functions


class FederatedAppServicer(functions_pb2_grpc.FederatedAppServicer):

    def InitilizeClients(self, request, context):
        response = functions_pb2.StatusPair()
        response.id = request.value
        response.status = functions.InitilizeClients()
        return response

    def SendModel(self, request, context):
        response = functions_pb2.Number()
        response.value = functions.SendModel(request.model)
        return response

    def Train(self, request, context):
        response = functions_pb2.TrainTriplet()
        response.id = request.id
        response.time = request.time
        response.model = functions.Train(request.model)
        return response
    def FetchClientResults(self, request, context):
        response = functions_pb2.Model()
        response.model = functions.FetchClientResults()
        return response

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

functions_pb2_grpc.add_FederatedAppServicer_to_server(FederatedAppServicer(), server)

print("Starting server on PORT: 8080")
# server.add_insecure_port('[::]:8080')
server.add_insecure_port('0.0.0.0:8080')
server.start()

try:
    while True:
        time.sleep(100)
except KeyboardInterrupt:
    server.stop(0)
