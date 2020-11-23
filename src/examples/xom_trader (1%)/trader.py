import requests
import time
import pickle
from datetime import date, datetime, timedelta
import os 
import threading
from multiprocessing import Pipe, Queue, Process, Pool
from trade_finder import signal_sender
#from test import sender
from iexfinance.stocks import Stock
import subtrader
import logging

global dir_path
dir_path = os.path.dirname(__file__)
logging.basicConfig(filename= '{}/trader.log'.format(dir_path), level=logging.DEBUG)
logger = logging.getLogger('trader')


# total_stop_loss = -.03 # 4%
# partial_stop_loss = -.01 #1%
# gain_stop = .015 #2%


def pipes():
    recv_conn, send_conn = Pipe()
    p = Process(target=signal_sender, args=((recv_conn, send_conn),))
    p.start()

    dir_path = os.path.dirname(__file__)
    tradefile = dir_path + '/tradeIDs'
    try:
        with open(tradefile, 'r') as file:

            lines = file.readlines()
            if lines == []:
                raise Exception

            last_tradeID = lines[-1].split(',')[0]
            tradeID = int(last_tradeID)
 
    except Exception as e:
        logger.error(str(e))
        tradeID = 0

    
    while True:
        
        logger.info('next tradeID: {}'.format(tradeID+1))
        receival = recv_conn.recv()
        logger.info('Signal recieval: {}'.format(receival))
        trade_signal = receival[0]
        pct_change = receival[1]


        if trade_signal == 'long':
            tradeID+=1 
            t1 = threading.Thread(target=subtrader.long, args=((tradeID,pct_change)))
            t1.start()
     
            logger.info('long order sent')
           
        
        if trade_signal == 'short':
            tradeID+=1 
            t1 = threading.Thread(target=subtrader.short, args=((tradeID,pct_change)))
            t1.start()
            logger.info('short order sent')
            

        if trade_signal == 'null':

            logger.info('no order sent')
            


        


if __name__=='__main__':
    pipes()


