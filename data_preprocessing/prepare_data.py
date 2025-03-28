import pandas as pd
import numpy as np
# import torch 
# import tensorflow as tf
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

        
        x,y=[],[]
        for i in gp:
            indx=gp[i].to_list()
            temp=self.raw_data.iloc[indx[0]:indx[-1]:self.time_window]
            temp=self.slide_it(temp)
            x.append(temp[0])
            y.append(temp[1])


        x=np.vstack(x)
        y=np.vstack(y)
        print(x.shape)
        print(y.shape)
        return 0

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
    
    # def get_data_torch(self):

    #     torch.tensor(se)




g={
    "data":"/mnt/6A8CB2D58CB29AD1/Project_live_trading/data/ADANIGREEN_minute.csv"
}
obj=PrepareData(g,3,4)
obj.get_data()


