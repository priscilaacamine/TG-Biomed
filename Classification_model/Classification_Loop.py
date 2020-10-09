#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importando as bibliotecas necessarias
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
from pickle import dump, load
import os, sys
# from imblearn.over_sampling import SMOTE
pd.set_option("display.max_rows", 300)


# In[2]:


class Net(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size):
        super(Net, self).__init__()
        self.num_layers = num_layers

        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size[0])])
        for i in range(0, self.num_layers-2):
            self.linears.extend([nn.Linear(layers_size[i], layers_size[i+1])])              
        self.linears.append(nn.Linear(layers_size[-1], output_size))

# Última camada sem função de ativação --> crossentropy já aplica softmax
# ReLU em intermediárias
    def forward(self, x):
        for layer in self.linears[0:-1]:                   
            x = F.relu(layer(x))
        x = (self.linears[-1](x))
        return(x)

# # Aplicando função de ativação na última camada tbm
# ### Tentar mudar pra sigmoide se deixar a normalização de 0,1
#     def forward(self, x):
#         for layer in self.linears:                   
#             x = torch.sigmoid(layer(x))
# #             x = F.relu(layer(x))
#         return(x)


# In[3]:


def nonRepeatedRandomInt(low, upper, N):
        import numpy as np
        import random
        
        numbers = np.arange(low, upper, 1)
        random.shuffle(numbers)
        shuffleNumbers = np.array(numbers)[0:int(N)]
                
        return shuffleNumbers

def createSurrogate(X):
    Xsur  = np.zeros_like(X)
    for i in range(X.shape[1]):
        Xsur[:,i] = X[nonRepeatedRandomInt(0, X.shape[0], X.shape[0]),i]
    return Xsur

def save_checkpoint(state, is_best, filename):
    if is_best:
        torch.save(state, filename)
#         print('*****Saved epoch: %d *****' % (state['epoch']))


# In[4]:


# ## Salva dado de treino/vali e dado de teste

# # Read clean data (features <10% excluded) = 25 features
# data = pd.read_pickle("data_closed_rigid_o_adjusted_clean")
# pf50 = np.round(data["PF50_closed_rigid"].values, 2)

# # Usar todas as 24 características + ones
# features = data.copy()
# features.drop("PF50_closed_rigid", axis=1, inplace=True)

# features_names = features.columns
# print('Features: ',list(features_names))
# X = features.values.astype('float')
# y = pf50.reshape(len(pf50), 1)

# # Separando o dataset em treino/vali e teste (treino/vali 70%, teste 30%)
# X_train_vali, X_test, y_train_vali, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Divide em classes de acordo com a os quartis do dado de treino/vali
# quart = np.quantile(y_train_vali,[0.25, 0.5, 0.75])
# print(quart)
# y_train_vali_class = np.digitize(y_train_vali, quart)
# y_test_class = np.digitize(y_test, quart)
# print((y_train_vali_class).shape)
# print((y_test_class).shape)

# # Salvando os dados em datasets separados

# np.save('X_test.npy', X_test)
# np.save('y_test.npy', y_test)
# np.save('y_test_class.npy', y_test_class)
# np.save('X_train_vali.npy', X_train_vali)
# np.save('y_train_vali.npy', y_train_vali)
# np.save('y_train_vali_class.npy', y_train_vali_class)


# In[5]:


# # Create empty xlsx with header
# tags = ["Net","random_state","num_layers","layers_size","net_param",
#         "criterion","learning_rate","optimizer","epochs",
#         "vali_best_epoch","vali_best_acc","vali_best_loss","vali_best_R",
#         "vali_sur_acc","vali_sur_loss","vali_sur_R","vali_c_matrix", 
#         "vali_c_matrix_perc", "test_acc","test_loss","test_R",
#         "test_predicted","test_c_matrix","test_c_matrix_perc"]
# df_nets = pd.DataFrame(columns = tags)
# print(df_nets)
# df_nets.to_excel ('classification_nets_empty.xlsx', index = False, header=True)


# In[6]:


# Loop de redes
# Load initial xlsx
df_nets = pd.read_excel('/home/priscila.a/TG-Biomed/Classification_model/classification_nets_empty.xlsx', index_col=None, header=0)

# Load data
X_train_vali = np.load('X_train_vali.npy')
y_train_vali_class = np.load('y_train_vali_class.npy')
X_test = np.load('X_test.npy')
y_test_class = np.load('y_test_class.npy')

if os.path.exists('Nets') == False: 
    os.makedirs('Nets')

for i in range(int(sys.argv[1]), int(sys.argv[2])):
    number_str = str(i)
    zero_filled_number = number_str.zfill(6)
    
    # Separando o treino da vali (treino 80%, validação 20%)
    # Deixar o high variável?
    random_state = np.random.randint(0, 10000, 1)[0]
#     print('Random State: %d' % (random_state))
    X_train, X_vali, y_train_class, y_vali_class = train_test_split(X_train_vali, y_train_vali_class, test_size=0.2, random_state=random_state)

    # Escalamento e Transformação dos dados
    scaler_x = RobustScaler(with_centering=True)
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_vali_scaled = scaler_x.transform(X_vali)
    X_test_scaled = scaler_x.transform(X_test)
    # save the scaler
    dump(scaler_x, open('./Nets/'+zero_filled_number+'_scaler_x.pkl', 'wb'))
    
    # Create Surrogate ## Para validação ou para teste?
    X_vali_scaled_sur = createSurrogate(X_vali_scaled)
    
    # Parâmetros da rede
    torch.manual_seed(1234)
    num_layers = 4
#     print('Number of layers: %d' % (num_layers))
    layer_init_size = np.random.randint(low=1, high=24)
#     print('Inicial size layer: %d' % (layer_init_size))
    layer_2_size = np.random.randint(low=1, high=24)
    layers_size = [layer_init_size, layer_2_size, 4]
#     print('Layers sizes:', list(layers_size))
    net = Net(input_size=X_train.shape[1], num_layers=num_layers, layers_size=layers_size , output_size=4)
    
    # Choose optmizer and loss function
    criterion = nn.CrossEntropyLoss()
    learning_rate = np.random.randint(low=1, high=100)/1000
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    
    # Treinamento 
    epochs = 50000
    loss_train = np.zeros(epochs)
    loss_vali = np.zeros(epochs)
    acc_vali = np.zeros(epochs)
    best_acc = 0
    
    for epoch in range(epochs):

        inputs = torch.autograd.Variable(torch.Tensor(X_train_scaled.astype(np.float32)).float())
        targets = torch.autograd.Variable(torch.Tensor(y_train_class).long())

        optimizer.zero_grad()
        out = net(inputs)
        loss = criterion(out, targets.squeeze())
        loss.backward()
        optimizer.step()

        loss_train[epoch] = loss.item()

        # Validação
        if epoch == 0 or (epoch + 1) % 100 == 0:
            inputs_vali = torch.autograd.Variable(torch.Tensor(X_vali_scaled.astype(np.float32)).float())
            targets_vali = torch.autograd.Variable(torch.Tensor(y_vali_class).long())
            out_vali = net(inputs_vali)
            loss_v = criterion(out_vali, targets_vali.squeeze())
            loss_vali[epoch] = loss_v.item()
            _, predicted = torch.max(out_vali.data, 1)

            # Calcula acurácia
            error_count = y_vali_class.size - np.count_nonzero((targets_vali.squeeze() == predicted) .numpy())
            acc_vali[epoch] = 100 * torch.sum(targets_vali.squeeze() == predicted) // y_vali_class.size
            
            r_vali = np.corrcoef(predicted.detach().numpy().squeeze(), targets_vali.detach().numpy().squeeze())[0,1]
            
            # remember best acc and save best model
            is_best = acc_vali[epoch] >= best_acc
            best_acc = max(acc_vali[epoch], best_acc)
            save_checkpoint({'epoch': epoch + 1,
                            #'arch': args.arch,
                            'state_dict': net.state_dict(),
                            'best_acc': best_acc,
                            'loss': loss_v.item(),
                            'R-corrcoef': r_vali,
                            'optimizer' : optimizer.state_dict(),
                            }, is_best, './Nets/'+zero_filled_number+'_model_best.pth.tar')

            if is_best:                
                inputs_vali_sur = torch.autograd.Variable(torch.Tensor(X_vali_scaled_sur.astype(np.float32)).float())
                targets_vali_sur = torch.autograd.Variable(torch.Tensor(y_vali_class).long())
                out_vali_sur = net(inputs_vali_sur)
                loss_v_sur = criterion(out_vali_sur, targets_vali_sur.squeeze())
                _, predicted_sur = torch.max(out_vali_sur.data, 1)
                
                # Calcula acurácia
                error_count_sur = y_vali_class.size - np.count_nonzero((targets_vali_sur.squeeze() == predicted_sur) .numpy())
                acc_vali_sur = 100 * torch.sum(targets_vali_sur.squeeze() == predicted_sur) // y_vali_class.size

                r_vali_sur = np.corrcoef(predicted_sur.detach().numpy().squeeze(), targets_vali_sur.detach().numpy().squeeze())[0,1]
                
                # Confusion matrix
                C_vali = confusion_matrix(targets_vali,predicted, labels=[0, 1, 2, 3])
                C_perc_vali = C_vali/np.sum(C_vali, axis=1, keepdims=True)*100
                
#             print('Epoch %d Loss: %.4f' % (epoch + 1, loss.item()))
#             print('   Validation Loss: %.4f' % (loss_v.item()))
#             print('   Errors: %d; Accuracy: %d%%' % (error_count, acc_vali[epoch]))
#             print('   R-corrcoef: %s' % (str(r_vali)))

    # Load best model
    checkpoint = torch.load('./Nets/'+zero_filled_number+'_model_best.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Teste
    # Avaliando a acurácia do modelo utilizando os dados de teste transformados
    inputs = torch.autograd.Variable(torch.Tensor(X_test_scaled.astype(np.float32)).float())
    targets = torch.autograd.Variable(torch.Tensor(y_test_class).long())

    optimizer.zero_grad()
    out = net(inputs)
    loss = criterion(out, targets.squeeze())
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(out.data, 1)       

    error_count = y_test_class.size - np.count_nonzero((targets.squeeze() == predicted) .numpy())
    acc = 100 * torch.sum(targets.squeeze() == predicted) //  y_test_class.size
    r = np.corrcoef(predicted.detach().numpy().squeeze(), targets.detach().numpy().squeeze())[0,1]

#     print('Errors: %d; Accuracy: %d%%' % (error_count, acc))
#     print('Teste Loss: %.4f' % (loss.item()))
#     print('R-corrcoef: %s' % (str(r)))

    # Confusion matrix
    C = confusion_matrix(targets,predicted, labels=[0, 1, 2, 3])
    C_perc = C/np.sum(C, axis=1, keepdims=True)*100

    net_info = {
            "Net": [zero_filled_number],
            "random_state": [random_state],
            "num_layers": [num_layers],
            "layers_size": [layers_size],
            "net_param": [net.parameters],
            "criterion": ['CrossEntropyLoss'],
            "learning_rate": [learning_rate],
            "optimizer": ['Adam'],
            "epochs": [epochs],
            "vali_best_epoch": [checkpoint['epoch']],
            "vali_best_acc": [checkpoint['best_acc']],
            "vali_best_loss": [checkpoint['loss']],
            "vali_best_R": [checkpoint['R-corrcoef']],
            "vali_sur_acc": [acc_vali_sur.item()],
            "vali_sur_loss": [loss_v_sur.item()],
            "vali_sur_R": [r_vali_sur],
            "vali_c_matrix": [C_vali],
            "vali_c_matrix_perc": [C_perc_vali],
            "test_acc": [acc.item()],
            "test_loss": [loss.item()],
            "test_R": [r],
            "test_predicted": [predicted.numpy()],
            "test_c_matrix": [C],
            "test_c_matrix_perc": [C_perc]
            }

    tags = ["Net","random_state","num_layers","layers_size","net_param",
            "criterion","learning_rate","optimizer","epochs",
            "vali_best_epoch","vali_best_acc","vali_best_loss","vali_best_R",
            "vali_sur_acc","vali_sur_loss","vali_sur_R","vali_c_matrix", 
            "vali_c_matrix_perc", "test_acc","test_loss","test_R",
            "test_predicted","test_c_matrix","test_c_matrix_perc"]
    df_nets = df_nets.append(pd.DataFrame(net_info, columns = tags), ignore_index=True)

    # Add suffix to identify saved info
    df_nets.to_excel ('/home/priscila.a/TG-Biomed/Classification_model/classification_nets_'+sys.argv[3]+'.xlsx', index = False, header=True)



