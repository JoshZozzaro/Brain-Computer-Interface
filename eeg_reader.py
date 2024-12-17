# -*- coding: utf8 -*-
#
#  CyKIT   2021.Nov.10
# ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
#  CyKIT.py 
#  Written by Warren
# 
#  Launcher to initiate EEG setup.
#  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯

import datetime # added by JZ 10/19/24
import os
import sys
import socket
import select
import struct
import eeg_output as eeg
import CyWebSocket
import threading
import time
import traceback
import inspect
    
def mirror(custom_string):
        try:
            print(str(datetime.datetime.now()) + str(custom_string))
            return
        except OSError as exp:
            return


def main(CyINIT):

    HOST = '127.0.0.1'    #str(sys.argv[1])
    PORT = 54123        #int(sys.argv[2])
    MODEL = 2           #int(sys.argv[3])
    check_connection = None
    parameters = 'outputdata' #str(sys.argv[4]).lower()

    #  Stage 1.
    # ¯¯¯¯¯¯¯¯¯¯¯
    #  Acquire I/O Object.
    # ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    cy_IO = eeg.ControllerIO()
    
    cy_IO.setInfo("ioObject", cy_IO)
    cy_IO.setInfo("config", parameters)   
    cy_IO.setInfo("verbose","False")

    if "noweb" in parameters:
        noweb = True
        cy_IO.setInfo("noweb","True")
        cy_IO.setInfo("status","True")
    else:
        noweb = False
        cy_IO.setInfo("noweb","False")
    
    if "noheader" in parameters:
        cy_IO.setInfo("noheader","True")

    headset = eeg.EEG(MODEL, cy_IO, parameters)
    
    while str(cy_IO.getInfo("DeviceObject")) == "0":
        time.sleep(.001)
        continue
    
    if "bluetooth" in parameters:
            mirror("> [Bluetooth] Pairing Device . . .")
    else:
        if "noweb" not in parameters:
            mirror("> Listening on " + HOST + " : " + str(PORT))

    mirror("> Trying Key Model #: " + str(MODEL))

    if "generic" in parameters:
        ioTHREAD = CyWebSocket.socketIO(PORT, 0, cy_IO)
    else:
        ioTHREAD = CyWebSocket.socketIO(PORT, 1, cy_IO)
    
    cy_IO.setServer(ioTHREAD)
    time.sleep(1)
    check_connection = ioTHREAD.Connect()
    ioTHREAD.start()
    
    while eval(cy_IO.getInfo("status")) != True:
        time.sleep(.001)
        continue   
    
    headset.start()
       
    CyINIT = 3
        
    while CyINIT > 2:
        
        CyINIT += 1
        time.sleep(.001)
        
        if (CyINIT % 10) == 0:
            

            check_threads = 0
            
            t_array = str(list(map(lambda x: x.name, threading.enumerate())))
            #if eval(cy_IO.getInfo("verbose")) == True:
            #    mirror(" Active Threads :{ " + str(t_array) + " } ")
            #time.sleep(15)
            
            if 'ioThread' in t_array:
                check_threads += 1
                
            if 'eegThread' in t_array:
                check_threads += 1

            if eval(cy_IO.getInfo("openvibe")) == True:
                if check_threads == 0:
                    ioTHREAD.onClose("CyKIT.main() 2")
                    mirror("\r\n*** Reseting . . .")
                    CyINIT = 1
                    main(1)
                continue
            
            #(1 if noweb == True else 2)
            
            if check_threads < (1 if noweb == True else 2):
                
                threadMax = 2
                totalTries = 0
                while threadMax > 1 and totalTries < 2:
                    totalTries += 1
                    time.sleep(0)
                    threadMax = 0
                    for t in threading.enumerate():
                        if "eegThread" in t.name:
                            cy_IO.setInfo("status","False")
                            #mirror(t.name)
                        if "ioThread" in t.name:
                            #mirror(t.name)
                            CyWebSocket.socketIO.stopThread(ioTHREAD)
                        
                        if "Thread-" in t.name:
                            #mirror(t.name)
                            threadMax += 1
                            try:
                                t.abort()
                            except:
                                continue
                t_array = str(list(map(lambda x: x.name, threading.enumerate())))
                #mirror(str(t_array))
                ioTHREAD.onClose("CyKIT.main() 1")
                mirror("*** Reseting . . .")
                CyINIT = 1
                main(1)

try:
    try:
        main(1)
    except OSError as exp:
        main(1)

except Exception as e:
    exc_type, ex, tb = sys.exc_info()
    imported_tb_info = traceback.extract_tb(tb)[-1]
    line_number = imported_tb_info[1]
    print_format = '{}: Exception in line: {}, message: {}'
    
    mirror("Error in file: " + str(tb.tb_frame.f_code.co_filename) + " >>> ")
    mirror("CyKITv2.Main() : " + print_format.format(exc_type.__name__, line_number, ex))
    mirror(traceback.format_exc())
    
    mirror(" ) WARNING_) CyKIT2.main E1: " + str(e))
    mirror("Error # " + str(list(OSError)))
    mirror("> Device Time Out or Disconnect . . .  [ Reconnect to Server. ]")
    main(1)
