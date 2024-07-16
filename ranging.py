import serial
import numpy as np
import matplotlib.pylab as plt
import torch

from sklearn.mixture import GaussianMixture
import torch.nn as nn

class RegressionNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RegressionNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.elu = nn.ELU()
            self.fc3 = nn.Linear(hidden_size, output_size)

        def forward(self, x): 
            out = self.fc1(x)
            out = self.elu(out)
            out = self.fc2(out)
            out = self.elu(out)
            out = self.fc3(out)
            return out
        
ser = serial.Serial('COM12', 115200)

times = 0
itration = 200
rawFrame = []
rtt_list = []
rssi_list = []

NN_with_GMM_model = torch.load('NN_with_GMM_model.pth')
NN_without_GMM_model = torch.load('NN_without_GMM_model.pth')
NN_RTT_model = torch.load('NN_RTT_model.pth')


        
def neural_network_predicting(model,test_x):
    # 在测试集上进行预测
    with torch.no_grad():
        predictions = model(test_x)

    predictions = predictions.numpy()
    return predictions

def GMM_filter(data):
    #best_aic,best_bic = compute_number_of_components(data,1,5)
    #n_components = best_aic  # 设置成分数量
    n_components = 2
    gmm = GaussianMixture(n_components=n_components)
    try:
        gmm.fit(data)
    except:
        lenghts = len(data)
        gmm.fit(data.reshape((lenghts,1)))
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    return means,covariances,weights


while True:
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

    if len(rtt_list) == 200:
        rtt_array = np.array(rtt_list) - 20074.659
        rssi_array = np.array(rssi_list)
        rtt_mean = np.mean(rtt_array)

        rtt_var = np.var(rtt_array)
        rssi_mean = np.mean(rssi_array)
        rssi_var = np.var(rssi_array)
        means,covariances,weights = GMM_filter(rtt_array)

        input_data = np.array([[float(means[0][0]),float(covariances[0][0]),float(means[1][0]),float(covariances[1][0]),
                                    float(weights[0]),float(weights[1]),
                                    rssi_mean,rssi_var,rtt_mean,rtt_var]])

        #input_data = torch.from_numpy(input_data)
        #print(input_data)
        test_x_RTT = torch.from_numpy(np.delete(input_data,[0,1,2,3,4,5,6,7],axis=1)).float()

        test_X_het = torch.from_numpy(input_data[:,6:]).float()

        test_X = torch.from_numpy(input_data[:,:]).float()

        NN_with_GMM_predictions = neural_network_predicting(NN_with_GMM_model,test_X)
        NN_without_GMM_predictions = neural_network_predicting(NN_without_GMM_model,test_X_het)
        NN_RTT_predictions = neural_network_predicting(NN_RTT_model,test_x_RTT)
        
        traditional_predictions = rtt_mean/2*299792458/16000000*0.4

        print('ML-based RTT+RSSI with GMM:',NN_with_GMM_predictions)
        print('ML-based RTT+RSSI:', NN_without_GMM_predictions)
        print('ML_based RTT:', NN_RTT_predictions)
        print('Traditional:', traditional_predictions)

        rtt_list = []
        rssi_list = []