import numpy
import torch
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


bounds = [[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],
          [-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0],[-1.0,1.0]
          ]
n_iter = 200
n_bits = 10
n_pop = 100
r_cross = 0.9
r_mut = 1.0/(float(n_bits)*len(bounds))   # or 0.01


EPOCH = 100
N_feature = 8
N_hidden = 20
N_prediction = 1
LR = 0.01



data = pd.read_excel('文件路径', header=None,index_col=None)
x = data.loc[:,0:12]
y = data.loc[:,13:13]

scaler = MinMaxScaler(feature_range=[0,1])
X = scaler.fit_transform(x)
Y = scaler.fit_transform(y)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
torch_dataset = torch.utils.data.TensorDataset(X,Y)
batch_size = 6


torch.manual_seed(1)
np.random.seed(1)

train , test_data = torch.utils.data.random_split(
    torch_dataset,
    [450,56]
)

train_data = torch.utils.data.DataLoader(
    train,
    batch_size=batch_size,
    shuffle = True
)



class BP_Net(nn.Module):
    def __init__(self, n_feature, n_output, n_neuron1, n_neuron2,n_layer):
        self.n_feature = n_feature
        self.n_output = n_output
        self.n_neuron1 = n_neuron1
        self.n_neuron2 = n_neuron2
        self.n_layer = n_layer
        super(BP_Net, self).__init__()
        self.input_layer = nn.Linear(self.n_feature,self.n_neuron1)
        self.hidden_layer = nn.Linear(self.n_neuron1,self.n_neuron2)
        self.output_layer = nn.Linear(self.n_neuron2,self.n_output)

    def forward(self, x):
        out = self.input_layer(x)
        out = torch.relu(out)
        out = self.hidden_layer(out)
        out = torch.relu(out)
        out = self.output_layer(out)
        return out

class GA:
    def __init__(self,bounds,n_bits,n_iter,n_pop,r_cross,r_mut):
        self.bounds = bounds
        self.n_bits = n_bits
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.r_cross = r_cross
        self.r_mut = r_mut

    def encode(self):

        self.pop = [np.random.randint(0, 2, self.n_bits*len(self.bounds)).tolist() for _ in range(self.n_pop)]

    def decode(self,bitstring):
        decoded = list()
        largest = len(self.bounds) ** self.n_bits
        for i in range(len(self.bounds)):
            start, end = i*self.n_bits, (i*self.n_bits) + self.n_bits
            substring = bitstring[start:end]
            chars = ''.join([str(s) for s in substring])
            integer = int(chars,2)

            value = self.bounds[i][0] + (integer/largest)*(self.bounds[i][1]-self.bounds[i][0])
            decoded.append(value)
        return decoded


    def selection(self,fitness,k=2):

        selection_ix = np.random.randint(len(self.pop))
        for ix in np.random.randint(0,len(self.pop),k):
            if fitness[ix] < fitness[selection_ix]:
                selection_ix = ix

        return self.pop[selection_ix]

    def crossover(self,p1,p2):

        c1,c2 = p1.copy(),p2.copy()

        if np.random.rand() < self.r_cross:
            pt = np.random.randint(1,len(p1)- 2)

            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]

        return [c1,c2]

    def mutation(self,bitstring):
        for i in range(len(bitstring)):
            if np.random.rand() < self.r_mut:
                bitstring[i] = 1 - bitstring[i]


    def runGA(self):
        self.encode()
        best, best_obj = 0, objective(self.decode(self.pop[0]))

        for gen in range(self.n_iter):
            decoded = [self.decode(p) for p in self.pop]
            fitness = [objective(d) for d in decoded]

            for i in range(self.n_pop):
                if fitness[i] < best_obj:
                    best, best_obj = self.pop[i],fitness[i]
                    print('>%d,new best f(%s) = %f' % (gen, decoded[i], fitness[i]))

            selected = [self.selection(fitness) for _ in range(self.n_pop)]

            children = list()

            for i in range(0,self.n_pop,2):
                p1,p2 = selected[i],selected[i+1]
                for c in self.crossover(p1,p2):
                    self.mutation(c)
                    children.append(c)

            self.pop = children

        return [best,best_obj]





BPModel = BP_Net(n_feature = N_feature,
                  n_output = N_prediction,
                  n_layer = 1,
                  n_neuron1 = N_hidden,
                  n_neuron2 = N_hidden)
print(BPModel)



criterion = nn.MSELoss()
optimizer = torch.optim.Adam(BPModel.parameters(), LR)


def objective(x):
    # BPModel.input_layer.weight     260   20×13
    # BPModel.input_layer.bias       20    20
    # BPModel.hidden_layer.weight    400   20×20
    # BPModel.hidden_layer.bias      20    20
    # BPModel.output_layer.weight    20    1×20
    # BPModel.output_layer.bias      1     1

    bp_wb = numpy.array(x)
    bp_input_w = bp_wb[0:260]
    np_bp_input_w = numpy.array(bp_input_w)
    input_w = np.zeros([20,13])
    for i in range(20):
        input_w[i][0:13] = np_bp_input_w[13 * i:13 * i + 13]
    bp_input_b = bp_wb[260:280]
    np_bp_input_b = numpy.array(bp_input_b)
    input_b = np.zeros([20])
    input_b[0:20] = np_bp_input_b
    torch_input_w = torch.Tensor(input_w)
    BPModel.input_layer.weight = torch.nn.Parameter(torch_input_w)
    torch_input_b = torch.Tensor(input_b)
    BPModel.input_layer.bias = torch.nn.Parameter(torch_input_b)

    bp_hidden_w = bp_wb[280:680]
    np_bp_hidden_w = numpy.array(bp_hidden_w)
    hidden_w = np.zeros([20,20])
    for i in range(20):
        hidden_w[i][0:20] = np_bp_hidden_w[20 * i:20 * i + 20]
    bp_hidden_b = bp_wb[680:700]
    np_bp_hidden_b = numpy.array(bp_hidden_b)
    hidden_b = np.zeros([20])
    hidden_b[0:20] = np_bp_hidden_b
    torch_hidden_w = torch.Tensor(hidden_w)
    BPModel.hidden_layer.weight = torch.nn.Parameter(torch_hidden_w)
    torch_hidden_b = torch.Tensor(hidden_b)
    BPModel.hidden_layer.bias = torch.nn.Parameter(torch_hidden_b)

    bp_output_w = bp_wb[700:720]
    np_bp_output_w = numpy.array(bp_output_w)
    output_w = np.zeros([1,20])
    output_w[0][0:20] = np_bp_output_w
    bp_output_b = bp_wb[720:721]
    np_bp_output_b = numpy.array(bp_output_b)
    output_b = np.zeros([1])
    output_b[0] = np_bp_output_b
    torch_output_w = torch.Tensor(output_w)
    BPModel.output_layer.weight = torch.nn.Parameter(torch_output_w)
    torch_output_b = torch.Tensor(output_b)
    BPModel.output_layer.bias = torch.nn.Parameter(torch_output_b)


    for epoch in range(EPOCH):
        BPModel.train()
        for batch_idx, (data, target) in enumerate(train_data):
            logits = BPModel.forward(data)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    prediction = []
    test_y = []
    BPModel.eval()
    for test_x, test_ys in test_data:
        predictions = BPModel(test_x)
        predictions = predictions.detach().numpy()
        prediction.append(predictions[0])
        test_ys.detach().numpy()
        test_y.append(test_ys[0])
    prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    test_y = scaler.inverse_transform(np.array(test_y).reshape(-1, 1))

    test_loss = criterion(torch.tensor(prediction, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
    print('MSE:', test_loss.detach().numpy())
    return test_loss.detach().numpy()




ga = GA(bounds,n_bits,n_iter,n_pop,r_cross,r_mut)
best,best_obj = ga.runGA()
print('Done!')
decoded = ga.decode(best)
print('objective(%s) = %f' % (decoded,best_obj))













