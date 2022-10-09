# import findspark
# findspark.init()

# import pyspark
# from pyspark import SparkContext
# from pyspark.streaming import StreamingContext
# from pyspark.sql import SQLContext
# from pyspark.sql.functions import desc


# sc = SparkContext()

# # we initiate the StreamingContext with 10 second batch interval. #next we initiate our sqlcontext
# ssc = StreamingContext(sc, 10)
# sqlContext = SQLContext(sc)

# # initiate streaming text from a TCP (socket) source:
# socket_stream = ssc.socketTextStream("127.0.0.1", 5555)

# # lines of tweets with socket_stream window of size 60, or 60 #seconds windows of time
# lines = socket_stream.window(60)

from socket import *
from select import *

HOST = '127.0.0.1'
PORT = 5555
ADDR = (HOST,PORT)

clientSocket = socket(AF_INET, SOCK_STREAM)

try:
    clientSocket.connect(ADDR)
    data = clientSocket.recv(65535)
    print('recieve data : ',data.decode())

except:
    pass