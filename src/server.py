from bs4 import BeautifulSoup 
import requests
import time
import pickle
from datetime import date, datetime, timedelta
import os
import traceback
import sys
dir_path = os.path.dirname(__file__)
timeformat = "%Y-%m-%d %H:%M:%S"
MST_to_EST = timedelta(hours = 2)
url = 'https://www.investing.com/commodities/crude-oil-historical-data'
headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36'}
base_hist = "https://oil-api.herokuapp.com/futures/historical"
base = "https://oil-api.herokuapp.com/futures"


def futures_data_reader(second_interval):

    while True:
        print('1')

        try:
            print('2t')
            source = requests.get(url, headers=headers, timeout=5)
            print('3t')
            source.encoding = 'utf-8' 
            print('4t')
            status = source.status_code
            print('5t')
            source = source.text
            print('6t')
            soup =  BeautifulSoup(source, 'lxml')
            print('7t')
            divs = soup.find("div", {"id": "quotes_summary_current_data"})
            print('t')
            price = divs.find('span').text
            print('9t')
            price = price.strip("'")
            print('10t')
            price = float(price)
            print('11t')
            now = (datetime.now() + MST_to_EST).strftime(timeformat)
            print('12t')
            print(now)
            print(price)
            print('13t')

            data_obj = {'time': now, 'close': price}


        except Exception as e:
            print(str(e))
            now = (datetime.now()+MST_to_EST).strftime(timeformat)
            data_obj = {'time': now, 'close': 'NaN'}
        try:
            print('14t')
            response = requests.post(base, data=data_obj, timeout=4)
            print('15t')
            print(response.text)
            print('16t')
        except:
            print('17e')
            print('timed out')

        print('18e')
        try:
            time.sleep(60)
        except Exception as e:
            print('excpetion:', str(e))
            print(traceback.print_exc())

if __name__ == '__main__':
    futures_data_reader(60)
