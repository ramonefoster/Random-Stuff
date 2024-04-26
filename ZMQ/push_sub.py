import zmq
import json
import time

# Para instalar a lib do zeroMQ, basta abrir o terminal 
# e digitar "pip install pyzmq"

# ADDRESS AND PORTA
ip_address = '127.0.0.1'
port_pub = 7005
port_pull = 7006
_msg_json = {
            "clientId": 66,
            "clientTransactionId": 0,
            "clientName": "Simulator",
            "action": "STATUS"
            }

# CONTEXT
context = zmq.Context()    

# SUBSCRIBER
subscriber = context.socket(zmq.SUB)
subscriber.connect(f"tcp://{ip_address}:{port_pub}")
topics_to_subscribe = ''
subscriber.setsockopt_string(zmq.SUBSCRIBE, topics_to_subscribe)

# PUSHER
pusher = context.socket(zmq.PUSH)
pusher.connect(f"tcp://{ip_address}:{port_pull}")

# POLL (Listen to multiple sockets and events)
poller = zmq.Poller()
poller.register(subscriber, zmq.POLLIN)

print("READY") # Everything OK

# SEND JSON
pusher.send_string(json.dumps(_msg_json))
time.sleep(1)
socks = dict(poller.poll(100))

# WAIT FOR PUB
if socks.get(subscriber) == zmq.POLLIN:
    message = subscriber.recv_string()
    print(message)

    