# -*- coding: utf-8 -*-
"""
This code reproduce the results of the numeical model data in Fig. 6
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt


# Create proability vectors for two frequencies
def Prob(w, g, delta_t, num_t, phi):
    # Ensure all arguments are numpy arrays or scalars
    num_t = np.asarray(num_t)
    phi = np.asarray(phi)
    P = np.square(np.cos(g * (np.sin(w*(num_t + 1)*delta_t + phi) - np.sin(w*(num_t)*delta_t + phi))/(2 * w) + 1*np.pi/4))
    return P

def ProbEff(w, g, delta_t, num_t, phi, eta):
    p = Prob(w, g, delta_t, num_t, phi)
    q = eta * p + 0.7 * eta * (1 - p)
    return q

def ProbV(w, g, delta_t, num_t, phi):
#    print(phi,num_t)
    P = np.square(np.cos(g * (np.sin(w*(num_t[np.newaxis,:] + 1)*delta_t + phi[:,np.newaxis]) - np.sin(w*(num_t[np.newaxis,:])*delta_t + phi[:,np.newaxis]))/(2 * w)+ 1*np.pi/4))
    return P

def ProbVEff(w, g, delta_t, num_t, phi, eta):
    p = ProbV(w, g, delta_t, num_t, phi)
    q = eta * p + 0.7 * eta * (1 - p)
    return q    

def MaxLike(y, P1n, P2n):
    x=torch.zeros(1,25000)
    x[0,:] = y
    P1 = torch.from_numpy(P1n).float()
    P2 = torch.from_numpy(P2n).float()
    p1v = x * P1 + (1-x) * (1 - P1 +  0.000001)
    p2v = x * P2 + (1-x) * (1 - P2 +  0.000001)
    L1 = torch.sum(np.log(p1v),1)
    L2 = torch.sum(np.log(p2v),1)
    M1 = torch.max(L1)
    M2 = torch.max(L2)
    if (M1 > M2):
        return 0
    else:
        return 1
    
    
dtype = torch.FloatTensor
freq_interval = 0.3             # frequenncy difference intervals
freq_init = 0.4                 # initial frequency difference
freq_num = 8                    # number of frequency difference points
average_num = 1                 # in case one is interested in averaging

freq_diff =np.zeros((1, freq_num))
freq_diff[0] =  np.arange(freq_init, freq_init + freq_num*freq_interval, freq_interval)
error_ml = np.zeros((1, freq_num))
error_lk = np.zeros((1, freq_num))
error_ml_min = np.ones((1, freq_num))
error_lk_min = np.ones((1, freq_num))

for df in range(freq_num):
    for k in range(average_num):
        print(df,k)
        data01mean, data02mean = 0.0635383, 0.0625315
        eta1, eta2 = (2.00/1.7) * data01mean, (2.00/1.7) * data02mean # 2.02
        w1, g1, g2, delta_t = 2*np.pi*250,  2*np.pi*12500, 2*np.pi*11250, 1e-5
        w2 = w1 + 2*np.pi*(freq_init + freq_interval * df)
        
        partition_fact = 4     # size reduction factor of the data sets        
        ndata = int(880*25000//partition_fact)
        
        data01n = np.ones((1,ndata), dtype=int)
        data02n = np.ones((1,ndata), dtype=int)    
        phi1, phi2 = torch.rand(1)[0] * 2 * np.pi, torch.rand(1)[0] * 2 * np.pi
        #generating the data
        for i in range(ndata):
            q1, q2 = torch.rand(1)[0], torch.rand(1)[0]
            if (q1 > ProbEff(w1, g1, delta_t, i, phi1, eta1)): data01n[0,i] = 0
            if (q2 > ProbEff(w2, g2, delta_t, i, phi2, eta2)): data02n[0,i] = 0

        
        class TrainDataset(Dataset):
            
            def __init__(self,jump,partition): # Jump - how many time points does one index skip
                ndata = int(880*25000//partition)
                data01 = data01n
                data02 = data02n
                ndatatr = int(0.7*ndata)
                data1tr = data01[0,0:ndatatr]
                data2tr = data02[0,0:ndatatr]
                self.x1 = torch.from_numpy(data1tr)
                self.x2 = torch.from_numpy(data2tr)
                self.y1 = 0
                self.y2 = 1
                self.len = 2*(data1tr.shape[0]-(25000))//jump 
                self.maxw1index = (data1tr.shape[0]-(25000))//jump 
                self.jump = jump
                
            def __getitem__(self,index):
                if index < self.maxw1index:
                  return self.x1[index*self.jump:index*self.jump+25000], self.y1
                elif index <= self.len:
                  return self.x2[(index-self.maxw1index)*self.jump:(index-self.maxw1index)*self.jump+25000], self.y2
                else:
                    raise StopIteration
    
            
            def __len__(self):
                return self.len
                
        class TestDataset(Dataset):
            
            def __init__(self,jump,partition):
                ndata = int(880*25000//partition)
                data01 =  data01n
                data02 =  data02n
                ndatatr = int(0.7*(ndata))
                data1ts = data01[0,ndatatr:ndata]
                data2ts = data02[0,ndatatr:ndata]
                self.x1 = torch.from_numpy(data1ts)
                self.x2 = torch.from_numpy(data2ts)
                self.len = 2*(data1ts.shape[0]-(25000))//jump  
                self.maxw1index = (data1ts.shape[0]-(25000))//jump 
                self.jump = jump
        
            def __getitem__(self,index):
                if index < self.maxw1index:
                  return self.x1[index*self.jump:index*self.jump+25000],0
                elif index <= self.len:
                  return self.x2[index*self.jump-self.maxw1index*self.jump:(index*self.jump-self.maxw1index*self.jump)+25000], 1
                else:
                    raise StopIteration
            def __len__(self):
                return self.len
                      

        jumpfac = 1
        batch_size = 50
        
        traindataset = TrainDataset(jumpfac,partition_fact)
        testdataset = TestDataset(jumpfac,partition_fact)
        print(" Size of testdataset : %f" % (testdataset.len)) 
        train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(testdataset, batch_size=1, num_workers=0)
 
      
        D_in, H, H2, D_out = 25000, 20, 35, 1 
        dtype = torch.FloatTensor
        loss_fn = torch.nn.MSELoss(reduction='mean')
        
#        ##################################################################################################
        # ML model
        model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H,H2),
                torch.nn.ReLU(),
                torch.nn.Linear(H2, D_out),
                )
        
        mintest = 1.
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
       
        for epoch in range(10):
            print("Train - Epoch  Loop %d" % epoch)
            for i, batch_data in enumerate(train_loader,0): 
                inputs, lables = batch_data
                inputs, lables = Variable(inputs.type(torch.FloatTensor)), Variable(lables.type(torch.FloatTensor))
                
                y_pred = model(inputs)
                loss = loss_fn(torch.sigmoid((y_pred[:,0]-0.5)), lables)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_pred2 = y_pred.clone()
            for j in range(y_pred2.shape[0]):
                if (y_pred2[j,0].item() > 0.5):
                    y_pred2[j,0] = 1
                else:
                    y_pred2[j,0] = 0
            loss1 = loss_fn(y_pred2[:,0], lables)
            print('Traing error at epoch (data)  %d is %f' % (epoch,loss1.item()))                
            
            avg_loss_tr = 0.0
            for i, batch_data in enumerate(test_loader,0):
                inputs, lables = batch_data
                inputs, lables = Variable(inputs.type(torch.FloatTensor)), Variable(lables.type(torch.FloatTensor))
                y_pred = torch.sigmoid((model.eval()(inputs)-0.5))
                y_pred2 = y_pred.clone()
                for j in range(y_pred2.shape[0]):
                    if (y_pred2[j].item() > 0.5):
                        y_pred2[j] = 1
                    else:
                        y_pred2[j] = 0
          
                loss2 = loss_fn(y_pred2[:,0], lables)
                avg_loss_tr += loss2.item()
            print('Test error at epoch %d is %f' % (epoch,avg_loss_tr/np.float(len(testdataset))  ))
            if avg_loss_tr/np.float(len(testdataset)) < mintest:
                mintest = avg_loss_tr/np.float(len(testdataset))
                print('best so far %f' % (mintest))
        print('Done. Best error is %f' % (mintest))

##################################################################################################
# Compare to Maximum likelihood
               
    
        # Compute prediction by maximun likelihood
        num_t_vec = np.arange(0, 25000, 1)
        phi_vec = np.arange(0, 2 * np.pi, 0.001 * 2 * np.pi)
        
        P_vec_w1 = ProbVEff(w1, g1, delta_t, num_t_vec, phi_vec, eta1) 
        P_vec_w2 = ProbVEff(w2, g2, delta_t, num_t_vec, phi_vec, eta2)
        
        loss3 = 0
        y_label = Variable(torch.zeros(1).type(dtype))
        y_pred_c = Variable(torch.zeros(1).type(dtype))
        for i in range(testdataset.len):
            y_label[0] = testdataset[i][1]
            m = MaxLike(testdataset[i][0], P_vec_w1, P_vec_w2)
            if m == 0:
                y_pred_c[0] = 0
            else:
                y_pred_c[0] = 1
            loss3 = loss3 + loss_fn(y_pred_c, y_label)

        LikeErr = loss3.item() / testdataset.len
        print(" likelihood error : %f" % (LikeErr))
 
        
        if (mintest < error_ml_min[0,df]):
            error_ml_min[0,df] = mintest
        if (LikeErr < error_lk_min[0,df]):
            error_lk_min[0,df] = LikeErr
            
        error_ml[0,df] = error_ml[0,df] + mintest
        error_lk[0,df] = error_lk[0,df] + LikeErr

        
error_ml = error_ml/average_num
error_lk = error_lk/average_num





