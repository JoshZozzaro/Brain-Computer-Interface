import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
import time
import pickle
import os

###############################################################################
#                                                                             #
#                                  -- LSTM CLASS --                           #
#                                                                             #
###############################################################################

# define the LSTM model class (same as the one used during training)
class LSTM(nn.Module):
    def __init__(self, no_feature=5, n_hidden=128, n_class=7):
        super(LSTM, self).__init__()

        self.lstm_layer = nn.LSTM(
            input_size=no_feature,
            hidden_size=n_hidden,         # LSTM hidden unit
            num_layers=2,           # number of LSTM layer
            bias=True,
            batch_first=True,       # input & output will have batch size as 1st dimension. e.g. (batch, segment_length, no_feature)
        )

        self.out = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm_layer(x.float(), None)
        r_out = F.dropout(r_out, 0.3)

        test_output = self.out(r_out[:, -1, :]) # choose r_out at the last time step
        return test_output

###############################################################################
#                                                                             #
#                                  -- ON STARTUP --                           #
#                                                                             #
###############################################################################

# ----------------------------------- init global variables ----------------- #

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
               'F4']

n_class = 7
no_feature = 5
segment_length = 8

# ------------------------------------ Hardware Check ----------------------- #

# check if a GPU is available
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

# ----------------------------------- Load Scaler data ---------------------- #

with open('loaded_scaler.pkl', 'rb') as f:
    scaler1 = pickle.load(f)

# ----------------------------------- Load model data ----------------------- #

# Load the trained model
model_path = 'lstm_model_v1.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM()
model.load_state_dict(torch.load(model_path, weights_only=False))   # explicitly set weights_only to False for future-proofing and to avoid warning message
model.to(device)
model.eval()

###############################################################################
#                                                                             #
#                                   -- MAIN LOOP --                           #
#                                                                             #
###############################################################################

input_directory = 'bci_processing'
sleep_time = 0.25
print("\nReady to receive data\n")

while True:

    dir = os.listdir(input_directory)

    while len(dir) == 0:
        #print("No files found")
        time.sleep(sleep_time)
        dir = os.listdir(input_directory)

    for file in dir:
        input_file = input_directory + "/" + file
        input_data = []

        #print(f"Reading file {input_file}")

        with open(input_file, 'r') as file:
            reader = csv.DictReader(file, delimiter=',', fieldnames = field_names)
            for line in reader:
                new_line = []
                new_line.append(int(line.get('F3')))
                new_line.append(int(line.get('FC5')))
                new_line.append(int(line.get('AF3')))
                new_line.append(int(line.get('F7')))
                new_line.append(int(line.get('P7')))
                input_data.append(new_line)

        dataset_1 = np.array(input_data)

        # ----------------------------------- Preprocessing ------------------------- #

        def extract(input, n_classes, n_fea, time_window, moving):
            xx = input[:, :n_fea]
            yy = input[:, n_fea:n_fea + 1]
            new_x = []
            new_y = []
            number = int((xx.shape[0] / moving) - 1)
            for i in range(number):
                ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
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
        #print('After segmentation, the shape of the data:', data_seg.shape)    # for debugging only

        no_longfeature = no_feature*segment_length
        data_seg_feature = data_seg[:, :no_longfeature]

        test_feature = data_seg_feature
        test_feature_2d = test_feature.reshape([-1, no_feature])

        test_fea_norm1 = scaler1.transform(test_feature_2d) # normalize the test data

        test_fea_norm1 = test_fea_norm1.reshape([-1, segment_length, no_feature])
        test_fea_norm1 = torch.tensor(test_fea_norm1).to(device)

        ###############################################################################
        #                                                                             #
        #                                   -- TEST --                                #
        #                                                                             #
        ###############################################################################

        output = []

        for sample in test_fea_norm1:

            # Reshape to 3D
            test_data_normalized = sample.reshape(-1, segment_length, no_feature)

            # Convert to PyTorch tensor
            test_tensor = torch.tensor(test_data_normalized).to(device)

            # Make predictions
            with torch.no_grad():
                predictions = model(test_tensor)
                predicted_labels = torch.max(predictions, 1)[1].cpu().numpy()

            #print("Predicted labels:", predicted_labels)   # for debugging only
            output.append(str(predicted_labels))

        print(f"Prediction for {input_file}: {max(set(output), key=output.count)}")
        #print(max(set(output), key=output.count))
        os.remove(input_file)
