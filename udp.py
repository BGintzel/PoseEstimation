import socket
import json
import time

########## send udp ########################
UDP_IP = "10.71.188.82"  #eduroam ip
UDP_IP = '127.0.0.1'     #localhost
UDP_PORT = 6789

MESSAGE_dict = {
    'fall_value':1,
    'confidence':0.8,
    'time_stamp':time.ctime(time.time())
}
MESSAGE_json = json.dumps(MESSAGE_dict).encode() #this is just a byte-string


print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE_json)
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.sendto(MESSAGE_json, (UDP_IP, UDP_PORT))
###############################################

########### receive udp #######################
UDP_IP = "10.71.188.82"
UDP_IP = '127.0.0.1'
UDP_PORT = 6789

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    message_json = data.decode()    #this is just a string
    message_dict = json.loads(message_json)
    print("received message: %s" % data)
