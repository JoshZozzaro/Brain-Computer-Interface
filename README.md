# Brain Computer Interface
This code is part of a project to construct a brain computer interface (BCI), which allows users to control computers using their thoughts. This project uses an Emotiv Epoc 1.0 EEG headset (an older, discontinued version of the [Epoc X](https://www.emotiv.com/collections/all/products/epoc-x))to detect the user's brain waves and uses [CyKit 3.0](https://github.com/CymatiCorp/CyKit) to connect to the headset. An LSTM machine learning model is then implemented in pytorch to recognize patterns in the user's brain waves.

The goal of the project is to:
- Record brain waves using an EEG sensor.
- Train a machine learning model to recognize patterns in EEG data as commands.
- Read live EEG data straight from the sensor into the trained machine learning model and recognize learned commands.

This repo in its current state is only a prototype, and is not yet fully functional. While all 3 goals are technically met, the Machine Learning model is extremely prone to overfitting due to a lack of proper signal processing applied to the recorded EEG signals. The model can get very good accuracy during training, but gets terrible results when trying to classify any new data it hasn't seen before. It is suspected that this is because (1) the model includes no preprocessing such as filtering or normalization, and (2) the training data is currently very limited. All training data used in this project has been recorded by myself to allow more control over the mental commands range of mental commands the system can identify. The end goal is that users will be able to easily record whatever mental commands are easiest for them and make the most sense for their application, and will know ahead of time roughly how much training data they will need to record in order to make the system work.

This repo is currently very messy, and the parts of the program that do work are very unoptimized. A complete refactor is currently in the works, but for noww, the project consists of 5 separate programs, rather than one program that can do everything, which is the end goal. As of right now, the following programs make up the software.
- CyKit 3.0 (the original version plus a modified version)
- minute_maker.py
- lstm_trainer.py
- data_halt.py
- real_time_lstm.py

## Part 1: Training the BCI

### CyKit 3.0
This software is based around [CyKit 3.0](https://github.com/CymatiCorp/CyKit) for connecting to the Emotiv headset. This project was not forked from CyKit as there is no intention of ever trying to merge the changes back into CyKit. CyKit is intended to give the user as many options as possible, While this project intends to remove as many options as possible, with the goal being to just connect a headset and immediately start working without worrying about any settings.

Recording is currently done completely in an umodified version of CyKit 3.0, which has not been included in this repo to reduce size. Using CYkit 3.0 with an Emotiv Epoc 1.0, the following stpes are taken:
- Start recording
- Perform mental command for about 90 seconds
- Stop recording and save as .csv file
- Repeat for all mental commands

### minute_maker.py
After recording, the individual csv files containing the training data need to be combined into a single file and labeled by class. Put all of the recorded .csv files into a directory and the run **minute_maker.py**. This script will:
- Ask for the path to the input directory with the csv files.
- Ask you to enter the name of the output file it will generate.
- Ask you to input a numerical class label for each input file
- Ask you to enter how many seconds into the recording the good data starts (it takes a few seconds to finish hitting record and start focusing on the metnal command you are trying to perform)
Once this information is entered, the program will pull 60 seconds worth of data from each input file (60 seconds starting from the time the used entered specifying when the "good" data started), append the numerical label to it, and append this to one file that contains all the training data. Note that this script does not currently check if there is 60 seconds worth of data available after the user-specified starting time, So you will need to verify this beforehand during recording.

### LSTM_trainer.py
This code is a modified version of pythonscripts/4-1-1_LSTM.py from Dr. Xiang Zhang's [Deep Learning for BCI](https://github.com/xiangzhang1015/Deep-Learning-for-BCI) repo from his and  Prof. Lina Yao's book [Deep Learning for EEG-based Brain-Computer Interface: Representations, Algorithms and Applications](https://www.amazon.com/Deep-Learning-EEG-Based-Brain-Computer-Interfaces/dp/1786349582). **This code was made available under the MIT License.** This code has been slightly modified from its original version to use csv files instead of npy files, and some other small changes such as the number of classes, segment length, and training epochs. To use this program, simply **modify line 34** to point to the location of the training data file constructed using minute_maker.py (the directory bci_training_data is intended for storing training data files), and then run the program. This will create a new version of loaded_scaler.pkl and lstm_model_v1.pth, which can then be loaded by real_time_lstm.py.

## Part 2: The real-time BCI Pipeline

To run the BCI system in real time, a modified bersion of CyKit is used to collect data and continuously write 1 second long csv files containing the EEG data to the directory bci_input_data. From here, the program **data_halt.py** is used to determine if the files should be deleted or moved into the directory bci_processing to be classified by the LSTM. This is a very crude method of allowing the datastream to be turned on and off while the EEG sensor is turned on. Finally, **real_time_lstm.py** loads the LSTM trained by lstm_trainer.py, feeds incoming data into it, and prints the classification to the console. 
