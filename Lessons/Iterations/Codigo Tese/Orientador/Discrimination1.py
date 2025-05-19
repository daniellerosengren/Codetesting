# -*- coding: utf-8 -*-
"""
This code reproduce the results of Fig. 4 and Fig. 5 
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def Prob(w, g, delta_t, num_t, phi, b):
    term1 = w * (num_t + 1) * delta_t + phi
    term2 = w * num_t * delta_t + phi
    sin_diff = np.sin(term1) - np.sin(term2)
    P = np.square(np.sin(g * (sin_diff / (2 * w)) + (delta_t / 2) * b + np.pi / 4))
    return P

def ProbEff(w, g, delta_t, num_t, phi, eta):
    p = Prob(w, g, delta_t, num_t, phi)
    q = eta * p + 0.7 * eta * (1 - p)
    return q

def ProbV(w, g, delta_t, num_t, phi):
    P = np.square(np.sin(g * (np.sin(w*(num_t[np.newaxis,:] + 1)*delta_t + phi[:,np.newaxis]) - np.sin(w*(num_t[np.newaxis,:])*delta_t + phi[:,np.newaxis]))/(2 * w) + 1*np.pi/4))
    return P

def ProbVEff(w, g, delta_t, num_t, phi, eta):
    p = ProbV(w, g, delta_t, num_t, phi)
    q = eta * p + 0.7 * eta * (1 - p)
    return q  


def MaxLike(x, P1n, P2n):
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
freq_interval = 0.0003          # frequency difference intervals
freq_init = 0.0003              # initial frequency difference
freq_num = 3                   # number of frequency difference points
average_num = 1                 # in case one is interested in averaging

freq_diff =np.zeros((1, freq_num))
freq_diff[0] =  np.arange(freq_init, (freq_num+1)* freq_interval, freq_interval)
error_ml = np.zeros((1, freq_num))
error_lk = np.zeros((1, freq_num))
error_lk2= np.zeros((1, freq_num))

num_t_vec = np.arange(0, 1000, 1)

for df in range(freq_num):
    for k in range(average_num):
        print(df,k)
        w1, g1, g2, delta_t = 10, 10, 10, 0.5
        w2 = w1 + freq_init + freq_interval * df

        ntrain = 125000
        ntest = 55000
        partition_fact = 1
        data01tr = np.zeros((ntrain,1000), dtype=int)
        data02tr = np.zeros((ntrain,1000), dtype=int)
        data01ts = np.zeros((ntest,1000), dtype=int)
        data02ts = np.zeros((ntest,1000), dtype=int)
        
        # Generating the data sets, currently the parameters correspond to Fig. 5.d
        for i in range(ntrain):
            n_b1, n_b2 = int(np.random.rand() * 1000), int(np.random.rand() * 1000)
            b1 = np.append(np.ones(n_b1) * np.random.normal(0,2), np.ones(1000 - n_b1) * np.random.normal(0,2))
            b2 = np.append(np.ones(n_b2) * np.random.normal(0,2), np.ones(1000 - n_b2) * np.random.normal(0,2))
            g1n = np.random.normal(g1,1*g1,1000)
            g2n = np.random.normal(g2,1*g2,1000)
            n_phi1, n_phi2 = int(np.random.rand() * 1000), int(np.random.rand() * 1000)
            phi1 = np.append(np.ones(n_phi1) * np.random.rand() * 2 * np.pi, np.ones(1000 - n_phi1) * np.random.rand() * 2 * np.pi)
            phi2 = np.append(np.ones(n_phi2) * np.random.rand() * 2 * np.pi, np.ones(1000 - n_phi2) * np.random.rand() * 2 * np.pi)
            xtr1 = np.random.rand(1000) > Prob(w1, g1n, delta_t, num_t_vec, phi1, b1) 
            xtr2 = np.random.rand(1000) > Prob(w2, g2n, delta_t, num_t_vec, phi2, b2) 
            data01tr[i,:] = xtr1.astype(int)
            data02tr[i,:] = xtr2.astype(int)

        for i in range(ntest):
            n_b1, n_b2 = int(np.random.rand() * 1000), int(np.random.rand() * 1000)
            b1 = np.append(np.ones(n_b1) * np.random.normal(0,2), np.ones(1000 - n_b1) * np.random.normal(0,2))
            b2 = np.append(np.ones(n_b2) * np.random.normal(0,2), np.ones(1000 - n_b2) * np.random.normal(0,2))
            g1n = np.random.normal(g1,1*g1,1000)
            g2n = np.random.normal(g2,1*g2,1000)
            n_phi1, n_phi2 = int(np.random.rand() * 1000), int(np.random.rand() * 1000)
            phi1 = np.append(np.ones(n_phi1) * np.random.rand(1) * 2 * np.pi, np.ones(1000 - n_phi1) * np.random.rand(1) * 2 * np.pi)
            phi2 = np.append(np.ones(n_phi2) * np.random.rand(1) * 2 * np.pi, np.ones(1000 - n_phi2) * np.random.rand(1) * 2 * np.pi)
            xts1 = np.random.rand(1000) > Prob(w1, g1n, delta_t, num_t_vec, phi1, b1) 
            xts2 = np.random.rand(1000) > Prob(w2, g2n, delta_t, num_t_vec, phi2, b2) 
            data01ts[i,:] = xts1.astype(int)
            data02ts[i,:] = xts2.astype(int)            
            

        
        class TrainDataset(Dataset):
            
            def __init__(self): 
                data1tr = data01tr
                data2tr = data02tr
                self.x1 = torch.from_numpy(data1tr)
                self.x2 = torch.from_numpy(data2tr)
                self.y1 = 0
                self.y2 = 1
                self.len = 2*(data1tr.shape[0]) 
                self.maxw1index = (data1tr.shape[0])            
                
            def __getitem__(self,index):
                if index < self.maxw1index:
                  return self.x1[index,:], self.y1
                elif index <= self.len:
                  return self.x2[(index-self.maxw1index),:], self.y2
                else:
                    raise StopIteration
                         
            
            def __len__(self):
                return self.len
                
        class TestDataset(Dataset):
            
            def __init__(self):        
                data1ts = data01ts
                data2ts = data02ts
                self.x1 = torch.from_numpy(data1ts)
                self.x2 = torch.from_numpy(data2ts)
                self.y1 = 0
                self.y2 = 1
                self.len = 2*(data1ts.shape[0])
                self.maxw1index = (data1ts.shape[0])                
        
            def __getitem__(self,index):
                if index < self.maxw1index:
                  return self.x1[index,:], self.y1
                elif index <= self.len:
                  return self.x2[index-self.maxw1index,:], self.y2
                else:
                    raise StopIteration
            def __len__(self):
                return self.len
                   
    
        batch_size = 50
        traindataset = TrainDataset()
        testdataset = TestDataset()
        
        train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(testdataset, batch_size=1, num_workers=0)
      
        D_in, H, H2, D_out = 1000, 20, 35, 1  
        dtype = torch.FloatTensor
        loss_fn = torch.nn.MSELoss(reduction='mean')
        
        ##################################################################################################
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
       
        for epoch in range(20):
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
            print('Test error at epoch %d is %f' % (epoch,avg_loss_tr/float(len(testdataset))  ))
            if avg_loss_tr/float(len(testdataset)) < mintest:
                mintest = avg_loss_tr/float(len(testdataset))
                print('best so far %f' % (mintest))
        print('Done. Best error is %f' % (mintest))


################################################################################################
# Compare to Maximum likelihood
    
        # Compute prediction by maximun likelihood
        num_t_vec = np.arange(0, 1000, 1)
        phi_vec = np.arange(0, 2 * np.pi, 0.01 * 2 * np.pi)
        
        P_vec_w1 = ProbV(w1, g1, delta_t, num_t_vec, phi_vec)
        P_vec_w2 = ProbV(w2, g2, delta_t, num_t_vec, phi_vec)
        
        loss3 = 0
        y_label =  Variable(torch.zeros(1).type(dtype))
        y_pred_l =  Variable(torch.zeros(1).type(dtype))
        for i in range(testdataset.len):
            y_label.data[0] = testdataset[i][1]
            y_pred_l.data[0] = 0
            a = testdataset[i][0].clone().type(dtype)
            a[a < 1] = 0
            m = MaxLike(a, P_vec_w1, P_vec_w2)
            if (m == 0):
               y_pred_l.data[0] = 0
            else:
               y_pred_l.data[0] = 1            
            loss3 = loss3 + loss_fn(y_pred_l, y_label)
            
        likeErr = loss3.data[0]/testdataset.len
        print(" Likelihood error: %s" % (likeErr)) 

        
###################################################################################################                        
#### Compare to Maximum likelihood by averaged correlations C(|i-j|)
        
          
        corrOrd = 700
        corrStart = 150
        cor_est1, cor_est2 = torch.zeros((1, corrOrd)), torch.zeros((1, corrOrd))
        for i in range(traindataset.maxw1index):
            a=traindataset[i][0].clone().type(dtype)
            a.resize_((1,D_in))
            b=traindataset[traindataset.maxw1index+i][0].clone().type(dtype)
            b.resize_((1,D_in))
            for j in range(corrOrd):
                ind = corrStart + j
                a1=a[0,0:D_in - (1+ind)]
                a2=a[0,(1+ind):D_in]
                cor_est1[0,j] =  cor_est1[0,j] + torch.mean((a1*a2 + 1)/2)
                b1=b[0,0:D_in - (1+ind)]
                b2=b[0,(1+ind):D_in]
                cor_est2[0,j] =  cor_est2[0,j] + torch.mean((b1*b2 + 1)/2)    
        
        
        cor_est1 = cor_est1/traindataset.maxw1index
        cor_est2 = cor_est2/traindataset.maxw1index
        
        cor_est1v = Variable(cor_est1)
        cor_est2v = Variable(cor_est2)
        
        loss3 = 0
        y_label =  Variable(torch.zeros(1).type(dtype))
        y_pred_c =  Variable(torch.zeros(1).type(dtype))
        for i in range(testdataset.len):
            if (i%10000 == 0):
                print(" testing, test number %d" % (i))
            y_label.data[0] = testdataset[i][1]
            y_pred_c.data[0] = 0
            a=testdataset[i][0].clone().type(dtype)
            a.resize_((1,D_in))
            cor_testv = Variable(torch.zeros((1, corrOrd)))
            for j in range(corrOrd):
                ind = corrStart + j
                a1=a[0,0:D_in - (1+ind)]
                a2=a[0,(1+ind):D_in]
                cor_testv[0,j].data[0] = torch.mean((a1*a2 + 1)/2)
        
            y_pred_c.data[0] = 0
            dis1 = torch.dist(cor_testv,cor_est1v)
            dis2 = torch.dist(cor_testv,cor_est2v)
            if (dis2.data[0] > dis1.data[0]):
                y_pred_c.data[0] = 0
            else:
                y_pred_c.data[0] = 1
            loss3 = loss3 + loss_fn(y_pred_c, y_label)
        
        corrErr = loss3.data[0]/testdataset.len
        print(" likelihood error : %f" % (corrErr))
        
            
        error_ml[0,df] = error_ml[0,df] + mintest
        error_lk[0,df] = error_lk[0,df] + corrErr
        error_lk2[0,df] = error_lk2[0,df] + likeErr

        
error_ml = error_ml/average_num
error_lk = error_lk/average_num
error_lk2 = error_lk2/average_num
 
plt.plot(freq_diff, error_ml, 'ro',  freq_diff,error_lk, 'gv', freq_diff, error_lk2, 'bs')
plt.show()

