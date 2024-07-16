import serial
import numpy as np
import matplotlib.pylab as plt

ser = serial.Serial('COM12', 115200)

times = 0
itration = 1000
rawFrame = []
rtt_list = []
rssi_list = []

while times < itration:
#while True:
    #while True:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-2:]==[13, 10]:
            if len(rawFrame) == 10:

                try:
                    rtt = int.from_bytes(rawFrame[0:4],byteorder='big')
                    #print('rtt:',rtt)
                    response_rssi = bytes(rawFrame[4:8])
                    response_rssi = int(response_rssi.decode('utf-8'))
                    #print('rssi:',response_rssi)
                    times = times + 1
                    rtt_list.append(rtt)
                    rssi_list.append(response_rssi)
                except:
                     rawFrame = []
            rawFrame = []

rtt_array = np.array(rtt_list)
rssi_array = np.array(rssi_list)

np.savez('data/distance1.npz', rtt = rtt_array, rssi = rssi_array)