import pandas as pd
import numpy as np
import torch 
import tensorflow as tf
import datetime

class PrepareData():

    def __init__(self,config,time_window,window):

        self.time_window=time_window
        self.window=window
        data_path=config['data']
        self.raw_data=pd.read_csv(data_path)

    def get_data(self):


        self.raw_data[['date',"time"]]=self.raw_data['date'].str.split(" ",expand=True)

        gp=self.raw_data.groupby("date").groups

        
        self.x,self.y=[],[]
        for i in gp:
            indx=gp[i].to_list()
            temp=self.raw_data.iloc[indx[0]:indx[-1]:self.time_window]
            temp=self.slide_it(temp)
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
    
    def get_torch_data(self):

        return torch.tensor(self.x,dtype=torch.float32,requires_grad=True),torch.tensor(self.y,dtype=torch.float32,requires_grad=True)
    
    def get_numpy_data(self):

        return self.x,self.y



