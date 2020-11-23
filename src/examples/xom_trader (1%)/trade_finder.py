
import requests
import os
import json 
import time
import pandas as pd
from multiprocessing import Pipe, Queue, Process
import logging
import datetime
import sys

global dir_path
dir_path = os.path.dirname(__file__)
logging.basicConfig(filename= '{}/trade_finder.log'.format(dir_path), level=logging.DEBUG)
logger = logging.getLogger('trade_finder')


base = "https://oil-api.herokuapp.com/futures/realtime"
#base = "https://oil-api.herokuapp.com/futures/historical"

global move_trigger
try:
    with open(dir_path+'/parameters', 'r') as file:
        
        lines = file.readlines()
        for line in lines:
            line = line.split('=')
            if line[0] == 'ticker':
                ticker = lines[-1]
                ticker = ticker.upper()
            if line[0] == 'move_trigger':
                move_trigger = float(line[-1])
except Exception as e:
    raise Exception('No "parameters" file')






# this function checks to see if it is a good time to check another set of data from
# crude oil. It only checks periodically becaause checking every minute will miss the big % moves!
# since big % moves don't happen in a minute. It also checks from previous close to new open.
def timeCheck(currentTime):
    times_start = [datetime.time(9, 30), datetime.time(9,59), datetime.time(10,59), 
                    datetime.time(11,59), datetime.time(12,59), datetime.time(13,59), 
                    datetime.time(14,59), datetime.time(15,59)]
    times_end = [datetime.time(9, 32), datetime.time(10, 1), datetime.time(11,1), 
                    datetime.time(12,1), datetime.time(13,1), datetime.time(14,1), 
                    datetime.time(15,1), datetime.time(16,1)]

    logger.info(currentTime)

    for i in range(len(times_start)):
        
        if (times_start[i] <=currentTime <= times_end[i]) == True:
            
            if (times_end[7] <= currentTime <= times_start[7]) == True:
                # if current time is between last close and market open, don't place a trade
                fillVar = 0
            else:
                print('it happened!')
                return True
                
        else:
            fillVar = 0




## Should have two functions: one for trading, one for analyzing the movement.
def signal_sender(pipe):

    recv_conn, send_conn = pipe

    # create close_data list that keeps a list of all data we gather
    close_data = []
    time_data = []

    # request data from API to initialize array, otherwise the loop won't have anything
    # to compare to 
    response = requests.get(base)
    response = response.json()
    close = response['close']
    times = response['time']
    
    # Append close and time data to respective lists
    close_data.append(close)
    time_data.append(times)



    while True:
        currentTime = datetime.datetime.fromtimestamp(time.time()+60*60*2).time()
        currentTime_forDataWriting = datetime.datetime.fromtimestamp(time.time()+60*60*2)
        currentTime_forDataWriting = currentTime_forDataWriting.strftime("%Y-%m-%d %H:%M:%S")
        print(currentTime_forDataWriting)
        
        
        tradeSignal = timeCheck(currentTime)
        logger.info(str(tradeSignal))

        if tradeSignal == True:

            try:
                response = requests.get(base)
                response = response.json()
                close = response['close']
                close_data.append(close)
                
                last_close = close_data[-2]
                cur_close = close_data[-1]


                pct_change = cur_close/last_close - 1
                logger.info('Pct change: {}'.format(pct_change))

                with open(dir_path+'/trade_finder', 'a') as file:
                    string_to_write = [currentTime_forDataWriting, str(pct_change)+'\n']
                    line = ','.join(string_to_write)
                    file.writelines(line)
                
                if abs(pct_change) > move_trigger:
                    logger.info('Moved above trigger: {}'.format(pct_change))
                    if pct_change > 0:
                        send_conn.send(['long', pct_change])  

                    if pct_change < 0:
                        send_conn.send(['short', pct_change])  

                else:
                    send_conn.send(['null', pct_change])
            except Exception as e:
                logger.error(str(e))

        else:
            logger.info('Not time.')
            
        
        time.sleep(60)
        
