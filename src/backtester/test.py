from multiprocessing import Process
import time
import datetime
import os
import requests
global dir_path
dir_path = os.path.dirname(__file__)
import traceback
print('hello')

x = 'ss'
print(int(x))
#print(traceback.print_exc())

# try:
#     with open(dir_path+'/parameters', 'r') as file:
        
#         lines = file.readlines()
#         for line in lines:
#             line = line.split('=')
#             if line[0] == 'ticker':
#                 ticker = lines[-1]
#                 ticker = ticker.upper()
#             if line[0] == 'move_trigger':
#                 move_trigger = int(line[-1])

        
# except Exception as e:
#     raise Exception('No "parameters" file')