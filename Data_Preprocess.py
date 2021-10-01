import numpy as np
import sys, os
import wfdb
import matplotlib.pyplot as plt 
import pywt
import pickle as pk
from collections import Counter
import time
import pandas as pd
start_time = time.time()

#The data names, beats and sub beats are annotated as prescribed by the authors of the database
#The link for the annotations - https://archive.physionet.org/physiobank/annotations.shtml

data_names = ['100', '101', '102', '103', '104', '105', '106', '107', 
              '108', '109', '111', '112', '113', '114', '115', '116', 
              '117', '118', '119', '121', '122', '123', '124', '200', 
              '201', '202', '203', '205', '207', '208', '209', '210', 
              '212', '213', '214', '215', '217', '219', '220', '221', 
              '222', '223', '228', '230', '231', '232', '233', '234']


types = ['N', 'S', 'V', 'F', 'Q']
beats = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
sub_beats = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N', 
           'A':'S', 'a':'S', 'J':'S', 'S':'S',
           'V':'V', 'E':'V',
           'F':'F',
           '/':'Q', 'f':'Q', 'Q':'Q'}
           
#Empty lists to store features and labels 
X = []
Y = []
num = 100

#----------------------------------------------------

#Denoising and Normalising to clean the data and extract 
#N, S, V, F, Q classes of signals
def denoise(sig, sigma, wn='bior1.3'): 
    #sig -> the input signal, sigma -> value for wavelet transform, wn -> wavelet transform family (biorthogonal in this case)
    threshold = sigma * np.sqrt(2*np.log2(len(sig)))
    c = pywt.wavedec(sig, wn) #signal decomposition (DWT)
    thresh = lambda x: pywt.threshold(x,threshold,'soft') #Soften the signals
    nc = list(map(thresh, c))
    return pywt.waverec(nc, wn) #signal reconstruction (IDWT)

#Parsing the previously downloaded data.
#There are multiple file for each record.
#Select the files with the .atr extension.

for d in data_names:
    #Parsing and extracting the data as prescribed in the official websie
    r=wfdb.rdrecord('./data/'+d)
    ann=wfdb.rdann('./data/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
    #Signals in file 114 are stored differently
    if d!='114':
        sig = np.array(r.p_signal[:,0])
    else:
        sig = np.array(r.p_signal[:,1])
    sig = denoise(sig, 0.005, 'bior1.3') #Calling the denoise function
    sig = (sig-min(sig)) / (max(sig)-min(sig)) #Normalising the signals between the lower and upper bound.

    
    # Extracting the features (signals) and labels (class for each signal)
    sig_len = len(sig)
    sym = ann.symbol
    pos = ann.sample
    beat_len = len(sym)
    for i in range(beat_len):
         #Since the file names start from 100
        if sym[i] in beats and pos[i]-num>=0 and pos[i]+num+1<=sig_len:
            a = sig[pos[i]-num:pos[i]+num+1]
            if len(a) != 2*num+1:
                print("Length error")
                continue
            X.append(a)
            Y.append(types.index(sub_beats[sym[i]]))

X = np.array(X) # Features data, containing the signals  
Y = np.array(Y) # Label data, containing the classes of the corresponding signals

print(Y)
print(X.shape)
print(Y.shape)
print(Counter(Y)) # counter is used to count the number of values in each class

#-----------------------------------------------------

#EDA and Visualising the extracted signals
#Find the number of records in each class of the heartbeats
values, counts = np.unique(Y, return_counts=True)
print(values, counts)
plt.figure(figsize=(20,10))
my_circle=plt.Circle( (0,0), 0.7, color = 'white')
plt.pie(counts, labels=['N', 'S', 'V', 'F', 'Q'], colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'],autopct='%2.2f%%', textprops={'fontsize': 18})
plt.title('Different classes of heartbeats', fontsize = 25)
p=plt.gcf()
#p.gca().add_artist(my_circle)

plt.show()

#------------------------------------------------------

#Plotting the graphs for each class of heartbeat
values, indexes = np.unique(Y, return_index=True)
for i in indexes:
    plt.plot(X[i])
    plt.xlabel("beat sample")
    plt.ylabel("time")
    if Y[i]==0:
        plt.title("Class N")
    elif Y[i]==1:
        plt.title("Class S")
    elif Y[i]==2:
        plt.title("Class V")
    elif Y[i]==3:
        plt.title("Class F")
    else:
        plt.title("Class Q")
    plt.show()
#--------------------------------------------------------

#Data to run ML algorithms on Spark
Features = pd.DataFrame(X)
#print(Features)

Labels = pd.DataFrame(Y)
#print(Labels)

ECG = pd.merge(Features, Labels, right_index = True, left_index = True)
#print(ECG)
ECD_Data = ECG.to_csv(r'ECG.csv', index=False)

#-----------------------------------------------------------

#Data to run ML algorithms locally
fn = "ECG_Data"+".pk"
with open(fn, "wb") as fw:
    pk.dump(X, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(Y, fw, protocol=pk.HIGHEST_PROTOCOL)

print("--- %s seconds ---" % (time.time() - start_time))
