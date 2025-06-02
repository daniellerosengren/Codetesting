# -*- coding: utf-8 -*-
"""
This code reproduces the results of $P_{M_{DL}}$ and $P_{M_{corr}}$ in Fig. 8
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import Dataset, DataLoader



dtype = torch.FloatTensor
freq_num = 1 
average_num = 1 

error_ml = np.zeros((1, freq_num))
error_lk = np.zeros((1, freq_num))
error_ml_min = np.ones((1, freq_num))
error_lk_min = np.ones((1, freq_num))

for df in range(freq_num):
    for k in range(average_num):
        print(df,k)
        
        partition_fact = 1
        Ndata = int(np.power(10,5)*256)    
        
        # Import the numerically generated data
        data01n = np.reshape(np.genfromtxt('ResData1.dat'),(1, Ndata))
        data02n = np.reshape(np.genfromtxt('ResData0d3.dat'),(1, Ndata))

        
        class TrainDataset(Dataset):
            
            def __init__(self,jump,partition): # Jump - how many time points does one index skip
                ndata = int(np.power(10,5)*256//partition)
                data01 = data01n
                data02 = data02n
                ndatatr = int(0.7*ndata)
                data1tr = data01[0,0:ndatatr]
                data2tr = data02[0,0:ndatatr]
                self.x1 = torch.from_numpy(data1tr)
                self.x2 = torch.from_numpy(data2tr)
                self.y1 = 0
                self.y2 = 1
                self.len = 2*(data1tr.shape[0]-(512))//jump 
                self.maxw1index = (data1tr.shape[0]-(512))//jump 
                self.jump = jump
                
            def __getitem__(self,index):
                if index < self.maxw1index:
                  return self.x1[index*self.jump:index*self.jump+512], self.y1
                elif index <= self.len:
                  return self.x2[(index-self.maxw1index)*self.jump:(index-self.maxw1index)*self.jump+512], self.y2
                else:
                    raise StopIteration    
            
            def __len__(self):
                return self.len
                
        class TestDataset(Dataset):
            
            def __init__(self,jump,partition):
                ndata = int(np.power(10,5)*256//partition)
                data01 =  data01n
                data02 =  data02n
                ndatatr = int(0.7*(ndata))
                data1ts = data01[0,ndatatr:ndata]
                data2ts = data02[0,ndatatr:ndata]
                self.x1 = torch.from_numpy(data1ts)
                self.x2 = torch.from_numpy(data2ts)
                self.len = 2*(data1ts.shape[0]-(512))//jump  
                self.maxw1index = (data1ts.shape[0]-(512))//jump 
                self.jump = jump
        
            def __getitem__(self,index):
                if index < self.maxw1index:
                  return self.x1[index*self.jump:index*self.jump+512],0
                elif index <= self.len:
                  return self.x2[index*self.jump-self.maxw1index*self.jump:(index*self.jump-self.maxw1index*self.jump)+512], 1
                else:
                    raise StopIteration
            def __len__(self):
                return self.len
               

        jumpfac = 1
        batch_size = 50
        
        traindataset = TrainDataset(jumpfac,partition_fact)
        testdataset = TestDataset(jumpfac,partition_fact)
        
        train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(testdataset, batch_size=1, num_workers=0)

      
        D_in, H, H2, D_out = 512, 20, 35, 1 
        dtype = torch.FloatTensor
        loss_fn = torch.nn.MSELoss(size_average=True)
        
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
                #epfac = 1. #20./epoch
                inputs, lables = batch_data
                inputs, lables = Variable(inputs.type(torch.FloatTensor)), Variable(lables.type(torch.FloatTensor))
                
                y_pred = model(inputs)
                loss = loss_fn(torch.sigmoid((y_pred[:,0]-0.5)), lables)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_pred2 = y_pred.clone()
            for j in range(y_pred2.shape[0]):
                if (y_pred2[j,0].data[0] > 0.5):
                    y_pred2[j,0] = 1
                else:
                    y_pred2[j,0] = 0
            loss1 = loss_fn(y_pred2[:,0], lables)
            print('Traing error at epoch (data)  %d is %f' % (epoch,loss1.data[0]))                
            
            avg_loss_tr = 0.0
            for i, batch_data in enumerate(test_loader,0):
                inputs, lables = batch_data
                inputs, lables = Variable(inputs.type(torch.FloatTensor)), Variable(lables.type(torch.FloatTensor))
                y_pred = torch.sigmoid((model.eval()(inputs)-0.5))
                y_pred2 = y_pred.clone()
                for j in range(y_pred2.shape[0]):
                    if (y_pred2[j].data[0] > 0.5):
                        y_pred2[j] = 1
                    else:
                        y_pred2[j] = 0
          
                loss2 = loss_fn(y_pred2[:,0], lables)
                avg_loss_tr += loss2.data[0]
            print('Test error at epoch %d is %f' % (epoch,avg_loss_tr/np.float(len(testdataset))  ))
            if avg_loss_tr/np.float(len(testdataset)) < mintest:
                mintest = avg_loss_tr/np.float(len(testdataset))
                print('best so far %f' % (mintest))
        print('Done. Best error is %f' % (mintest))


###################################################################################################                        
### Compare to Maximum likelihood by averaged correlations C(|i-j|)
        
          
        corrOrd = 510
        corrStart = 1
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
        









