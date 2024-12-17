# MIT License
# 
# Copyright (c) 2022 Xiang Zhang
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report #, confusion_matrix
import csv

input_file = "bci_training_data/7_class_12-15-24.csv"

input_data = []
field_names = ['F3',
               'FC5',
               'AF3',
               'F7',
               'T7',
               'P7',
               'O1',
               'O2',
               'P8',
               'T8',
               'F8',
               'AF4',
               'FC6',
               'F4',
               'label']

with open(input_file, 'r') as file:
    reader = csv.DictReader(file, delimiter=',', fieldnames = field_names)
    for line in reader:
        new_line = []
        new_line.append(int(line.get('F3')))
        new_line.append(int(line.get('FC5')))
        new_line.append(int(line.get('AF3')))
        new_line.append(int(line.get('F7')))
        new_line.append(int(line.get('P7')))
        new_line.append(int(line.get('label')))
        input_data.append(new_line)

dataset_1 = np.array(input_data)
print('dataset_1 shape:', dataset_1.shape)

# check if a GPU is available
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

n_class = 7         # number of classes (0-3)
no_feature = 5     # dimension of vectors
segment_length = 8 # 128 / 8 = 16

#LR = 0.005  # learning rate
LR = 0.005
EPOCH = 751
n_hidden = (64 * 2) # number of neurons in hidden layer
l2 = 0.001  # the coefficient of l2-norm regularization
test_percent = 0.5

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    y_ = [int(xx) for xx in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def extract(input, n_classes, n_fea, time_window, moving):
    xx = input[:, :n_fea]                          # get all rows and all columns up to but NOT including column n_fea
    yy = input[:, n_fea:n_fea + 1]                 # get all rows but only column n_fea
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)    # calculate total number of windows that can be created given moving step size
    for i in range(number):                     # for every possible window that can be created
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])   # calculate average value of label
        if ave_y in range(n_classes + 1):
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    data = np.vstack((data, data[-1]))  # add the last sample again, to make the sample number round
    return data

data_seg = extract(dataset_1, n_classes=n_class, n_fea=no_feature, time_window=segment_length, moving=(segment_length/2))  # 50% overlapping
print('After segmentation, the shape of the data:', data_seg.shape)

# split training and test data
no_longfeature = no_feature*segment_length
data_seg_feature = data_seg[:, :no_longfeature]
data_seg_label = data_seg[:, no_longfeature:no_longfeature+1]
train_feature, test_feature, train_label, test_label = train_test_split(data_seg_feature, data_seg_label,test_size=test_percent, shuffle=True)

# normalization
# before normalize reshape data back to raw data shape
train_feature_2d = train_feature.reshape([-1, no_feature])
test_feature_2d = test_feature.reshape([-1, no_feature])

scaler1 = StandardScaler().fit(train_feature_2d)
train_fea_norm1 = scaler1.transform(train_feature_2d) # normalize the training data
test_fea_norm1 = scaler1.transform(test_feature_2d) # normalize the test data
print('After normalization, the shape of training feature:', train_fea_norm1.shape,
      '\nAfter normalization, the shape of test feature:', test_fea_norm1.shape)

# after normalization, reshape data to 3d in order to feed in to LSTM
train_fea_norm1 = train_fea_norm1.reshape([-1, segment_length, no_feature])
test_fea_norm1 = test_fea_norm1.reshape([-1, segment_length, no_feature])
print('After reshape, the shape of training feature:', train_fea_norm1.shape,
      '\nAfter reshape, the shape of test feature:', test_fea_norm1.shape)

BATCH_size = int(test_fea_norm1.shape[0]) # use test_data as batch size

# feed data into dataloader
train_fea_norm1 = torch.tensor(train_fea_norm1).to(device)
train_label = torch.tensor(train_label.flatten()).to(device)
train_data = Data.TensorDataset(train_fea_norm1, train_label)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_size, shuffle=False)

test_fea_norm1 = torch.tensor(test_fea_norm1).to(device)
test_label = torch.tensor(test_label.flatten()).to(device)

# classifier
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm_layer = nn.LSTM(
            input_size=no_feature,
            hidden_size=n_hidden,         # LSTM hidden unit
            num_layers=2,           # number of LSTM layer
            bias=True,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, segment_length, no_feature)
        )

        self.out = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm_layer(x.float(), None)
        r_out = F.dropout(r_out, 0.3)                           # 30% of neurons are randomly set to zero during training

        test_output = self.out(r_out[:, -1, :]) # choose r_out at the last time step
        return test_output

lstm = LSTM()
lstm.to(device)
print(lstm)

optimizer = torch.optim.Adam(lstm.parameters(), lr=LR, weight_decay=l2)   # optimize all parameters
loss_func = nn.CrossEntropyLoss()

best_acc = []
best_auc = []

# training and testing
start_time = time.perf_counter()
for epoch in range(EPOCH):
    for step, (train_x, train_y) in enumerate(train_loader):

        output = lstm(train_x)  # LSTM output of training data
        loss = loss_func(output, train_y.long())  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    if epoch % 10 == 0:
        test_output = lstm(test_fea_norm1)  # LSTM output of test data
        test_loss = loss_func(test_output, test_label.long())

        test_y_score = one_hot(test_label.data.cpu().numpy())  # .cpu() can be removed if your device is cpu.
        pred_score = F.softmax(test_output, dim=1).data.cpu().numpy()  # normalize the output
        auc_score = roc_auc_score(test_y_score, pred_score)

        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        pred_train = torch.max(output, 1)[1].data.cpu().numpy()

        test_acc = accuracy_score(test_label.data.cpu().numpy(), pred_y)
        train_acc = accuracy_score(train_y.data.cpu().numpy(), pred_train)

        print('Epoch: ', epoch, '|train loss: %.4f' % loss.item(),
              ' train ACC: %.4f' % train_acc, '| test loss: %.4f' % test_loss.item(),
              'test ACC: %.4f' % test_acc, '| AUC: %.4f' % auc_score)
        best_acc.append(test_acc)
        best_auc.append(auc_score)

current_time = time.perf_counter()
running_time = current_time - start_time
print(classification_report(test_label.data.cpu().numpy(), pred_y))
print('BEST TEST ACC: {}, AUC: {}'.format(max(best_acc), max(best_auc)))
print("Total Running Time: {} seconds".format(round(running_time, 2)))

torch.save(lstm.state_dict(), 'lstm_model_v1.pth')

#confusion_matrix(train_y, pred_y, *, labels=None, sample_weight=None, normalize=None) doesn't work yet
