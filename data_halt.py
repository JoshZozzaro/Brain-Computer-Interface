# Created by Joshua Zozzaro 12/1/24

import os
import threading
from time import sleep

halt = True
exit = False

input_directory = 'bci_data_transit'
output_directory = 'bci_processing'

def data_halt():
    global halt
    global exit
    while True:
        if exit:
            break
        if halt:
            for file in os.listdir(input_directory):
                os.remove(input_directory + "/" + file)
        else:
            for file in os.listdir(input_directory):
                os.rename(input_directory + "/" + file, output_directory + "/" + file)
        sleep(0.1)

print("Program Started")
data_halt_thread = threading.Thread(target=data_halt)
data_halt_thread.start()

while True:
    sleep(0.1)
    print("'H' -----> Halt Data")
    print("'Enter' -> Pass Data")
    print("'Q' -----> Quit Program")
    control = input("Enter selection : ")
    if 'h' in control.lower():
        halt = True
        print("\n\n\nHalting data\n")
    elif 'q' in control.lower():
        print("\n\n\nQuitting program...\n")
        exit = True
        data_halt_thread.join()
        quit()
    else:
        halt = False
        print("\n\n\nAllowing data to pass\n")
