import pandas as pd
import numpy as np
import torch 
# import tensorflow as tf
from torch.utils.data import DataLoader,Dataset,random_split

class PrepareData():

    def __init__(self,config,time_window,window):

        self.time_window=time_window
        self.window=window
        data_path=config['data']
        self.raw_data=pd.read_csv(data_path)
        self.x,self.y=[],[]

    def get_data(self):


        self.raw_data[['date',"time"]]=self.raw_data['date'].str.split(" ",expand=True)

        gp=self.raw_data.groupby("date").groups

        
        
        for i in gp:
            indx=gp[i].to_list()
            temp=self.raw_data.iloc[indx[0]:indx[-1]:self.time_window]
            temp=self.slide_it(temp)
            if len(temp[0])!=0:
                self.x.append(temp[0])
                self.y.append(temp[1])

        self.x=np.vstack(self.x)
        self.y=np.vstack(self.y)

    def slide_it(self,x):   

        temp_x=[]
        temp_y=[]
        x=x['close'].to_numpy()
        for i in range(len(x)//self.window-self.window):
            temp_x.append(x[i:i+self.window])
            temp_y.append(x[i+self.window])
        temp_x=np.array(temp_x)
        temp_y=np.array(temp_y)
        temp_y=temp_y.reshape(-1,1)
        return temp_x,temp_y
    
    def get_torch_data(self,batch_size):
        
        self.get_data()
        dataset=MyDatset(self.x,self.y)
        train_dataset,test_dataset=random_split(dataset,[.8,.2])
        train_dataloader=DataLoader(train_dataset,batch_size)   
        test_dataloader=DataLoader(test_dataset,batch_size)   

        return train_dataloader,test_dataloader
    
    def get_numpy_data(self):

        return self.x,self.y



        

class MyDatset(Dataset):
    def __init__(self,x,y):

        self.x=torch.tensor(x,dtype=torch.float32,requires_grad=True)
        self.y=torch.tensor(y,dtype=torch.float32,requires_grad=True)
            
    def __len__(self):

        return len(self.x)

    def __getitem__(self, index):
        return self.x[index],self.y[index]    