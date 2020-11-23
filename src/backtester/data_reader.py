from bs4 import BeautifulSoup 
import requests
import time
import pickle
from datetime import date, datetime, timedelta


dir_path = os.path.dirname(__file__)
timeformat = "%Y-%m-%d %H:%M:%S"
MST_to_EST = timedelta(hours = 2)
url = 'https://www.investing.com/commodities/crude-oil-historical-data'
headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36'}

def futures_data_reader(second_interval):

    while True:
        try:
            source = requests.get(url, headers=headers, timeout = 1.5)
            source.encoding = 'utf-8' 
            status = source.status_code
            source = source.text
            soup =  BeautifulSoup(source, 'lxml')
            divs = soup.find("div", {"id": "quotes_summary_current_data"})
            price = divs.find('span').text
            price = price.strip("'")
            price = float(price)
            now = (datetime.now() + MST_to_EST).strftime(timeformat)
            line = '{},{}\n'.format(now, price)


        except Exception as e:
            print(str(e))
            now = (datetime.now() + MST_to_EST).strftime(timeformat)
            line = '{},{}\n'.format(now, 'NaN')
            
        with open('./futures_data_scraper.txt', 'a') as file:
            file.write(line)

        

        time.sleep(second_interval)
    child_conn.close()

if __name__ == '__main__':
    futures_data_reader(60)