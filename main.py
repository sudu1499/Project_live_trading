from data_preprocessing.prepare_data import PrepareData
from torch_model.model import LiveTrading
from tqdm import tqdm
import json
import torch
import matplotlib.pyplot as plt

config=json.load(open("/mnt/6A8CB2D58CB29AD1/Project_live_trading/config.json",'r'))

data=PrepareData(config,5,5)

train_data,test_data=data.get_torch_data(32)

lstm_input_size=next(iter(train_data))[0].shape[1]
lstm_hidden_size=20
num_layer=2
bidirectional=True
linear_hidden_size=200
epochs=50
model=LiveTrading(lstm_input_size,lstm_hidden_size,num_layer,bidirectional,linear_hidden_size).to("cuda")

loss_fn=torch.nn.MSELoss().to("cuda")
opt=torch.optim.SGD(model.parameters())

for e in tqdm(range(epochs)):
    total_loss=[]

    for i in train_data:

        ypred=model(i[0].to("cuda")).to("cuda")
        loss=loss_fn(ypred,i[1].to("cuda"))
        total_loss.append(loss.to("cpu").detach().numpy())

        opt.zero_grad()
        loss.backward()
        opt.step()
        

plt.plot(total_loss)
plt.show()


# import torch
# print('CUDA:',torch.version.cuda)

# cudnn = torch.backends.cudnn.version()
# cudnn_major = cudnn // 1000
# cudnn = cudnn % 1000
# cudnn_minor = cudnn // 100
# cudnn_patch = cudnn % 100
# print( 'cuDNN:', '.'.join([str(cudnn_major),str(cudnn_minor),str(cudnn_patch)]) )