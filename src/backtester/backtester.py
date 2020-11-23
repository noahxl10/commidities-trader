import pandas as pd 
from datetime import date, datetime, timedelta
import os
import pickle
import logging
import time

from pandas.io.parsers import read_csv
import plotly.express as px
import matplotlib.pyplot as plt
import finplot as fplt
import numpy as np
import seaborn as sns
from scipy import stats
import cProfile, pstats, io
from pstats import SortKey
pd.set_option('display.max_columns', None)


def dir_path():
    return os.path.dirname(__file__)


def xom_dir():
    return dir_path() + '/raw/xom'


def cl_dir():
    return dir_path() + '/raw/cl'


def pickle_path():
    return dir_path() + '/pickles'


def datetime_format():
    return "%Y-%m-%d %H:%M:%S"


def xom_paths(type_of_file):

    if type_of_file == 'csv':
        file_1 = xom_dir() + '/XOM_2000_2009.csv'
        file_2 = xom_dir() + '/XOM_2010_2019.csv'
        file_3 = xom_dir() + '/XOM_2020_2020.csv'
        return [file_1, file_2, file_3]

    if type_of_file == 'txt':
        file_1 = xom_dir() + '/XOM_2000_2009.txt'
        file_2 = xom_dir() + '/XOM_2010_2019.txt'
        file_3 = xom_dir() + '/XOM_2020_2020.txt'
        return [file_1, file_2, file_3]

    else:
        raise Exception("Please input 'txt' or 'csv' ")


def cl_paths(type_of_file):
    
    if type_of_file == 'csv':
        file_1 = cl_dir() + '/CL_2000_2009.csv'
        file_2 = cl_dir() + '/CL_2010_2019.csv'
        file_3 = cl_dir() + '/CL_2020_2020.csv'
        return [file_1, file_2, file_3]

    if type_of_file == 'txt':
        file_1 = cl_dir() + '/CL_2000_2009.txt'
        file_2 = cl_dir() + '/CL_2010_2019.txt'
        file_3 = cl_dir() + '/CL_2020_2020.txt'
        return [file_1, file_2, file_3]

    else:
        raise Exception("Please input 'txt' or 'csv' ")



def txt_to_dataframe(file_path):
    with open(file_path, 'r') as file:

        lines = file.readlines()

        split_lines = [line.replace('\n', '').split(',') for line in lines]

        date_time = [row[0] for row in split_lines]
        open_price = [row[1] for row in split_lines]
        high_price = [row[2] for row in split_lines]
        low_price = [row[3] for row in split_lines]
        close_price = [row[4] for row in split_lines]
        volume = [row[5] for row in split_lines]

    temp_dict = {'time' : date_time,
                 'open' : open_price,
                 'high' : high_price,
                 'low' : low_price,
                 'close' : close_price,
                 'volume' : volume}
    
    df = pd.DataFrame.from_dict(temp_dict)

    return df


def csv_to_dataframe(file_path):

    columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(file_path, names = columns)

    df['time'] = pd.to_datetime(df['time'], format=datetime_format())

    return df


def combine_all_data(file_paths):
    dataframes = [csv_to_dataframe(file_path) for file_path in file_paths]

    try:
        final_df = dataframes[0]
        for dataframe in dataframes[1:]:
            final_df = final_df.append(dataframe)

    except Exception as e:
        print(str(e))

    return final_df


def filter_preCL_dates(cl_df, stock_df):

    first_cl_time = datetime.strftime(cl_df.time.iloc[0], datetime_format())

    stock_df = stock_df[ stock_df['time'] > first_cl_time]
    stock_df = stock_df.reset_index().drop(columns = ['index'])

    return stock_df




def get_final_dataframe(name_):

    if name_ in ['xom', 'XOM']:
        print('Getting data for XOM')
        path  = pickle_path() + '/xom.pkl'
        try: 
            final_df = pd.read_pickle(path)
        except Exception as e:
            print(str(e))
            final_df = combine_all_data(xom_paths('csv'))
            final_df.to_pickle(path)


        return final_df.reset_index().drop(columns=['index'])


    if name_ in ['CL', 'cl', 'oil', 'futures']:
        print('Getting data for oil futures')
        path = pickle_path() + '/cl.pkl'
        try: 
            final_df = pd.read_pickle(path)
        except Exception as e:
            print(str(e))
            
            final_df = combine_all_data(cl_paths('csv'))
            final_df.to_pickle(path)



        return final_df.reset_index().drop(columns=['index'])


def plot_price(df):
    df = df.set_index('time')
    fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']])
    fplt.show()


def plot_price_old(df):
    graph_df = df[['time', 'close']].copy()
    graph_df.plot(x = 'time', y = 'close')
    plt.title('Plot of {} dataset'.format(''))
    plt.show()


def date_filter(df, date_string):
    # year-month-day
    #ex: 2020-05-25

    time_filter = '{} 00:00:00'.format(date_string)

    filtered_df = df[ df['time'] > time_filter]
    filtered_df = filtered_df.reset_index().drop(columns = ['index'])
    return filtered_df 



def find_specific_performance(cl_df, stock_df, performance_benchmark):
    
    ## e.g. I want to find all times where there was a greater than 5% return
    ## or a less than 5% loss from the previous period
    performance_benchmark = performance_benchmark

    cl_df['close_pct_change'] = cl_df['close'].pct_change() #.abs()

    stock_df['close_pct_change'] = stock_df['close'].pct_change() #.abs()

    cl_df_1 = cl_df[ cl_df['close_pct_change'] < -performance_benchmark]*100
    cl_df_2 = cl_df[ cl_df['close_pct_change'] > performance_benchmark]*100
    cl_df = cl_df_1.append(cl_df_2)
    cl_df = cl_df.sort_index()

    stock_df_1 = stock_df[ stock_df['close_pct_change'] < -performance_benchmark]*100
    stock_df_2 = stock_df[ stock_df['close_pct_change'] > performance_benchmark]*100
    stock_df = stock_df_1.append(stock_df_2)
    stock_df = stock_df.sort_index()

    return  cl_df, stock_df

    
def resample(df, timeframe):
    # 15T means 15 buckets of whatever the time interval is, which is in minutes in this case
    # so 15T is 15 minutes...60T would be 60 minutes
    # timeseries = xom_final_df.resample('15T', on='time').close.last() #can do .mean() or .sum()
    new_df = pd.DataFrame()
    new_df['open'] = df.resample(timeframe, on='time').open.last()
    new_df['high'] = df.resample(timeframe, on='time').high.last()
    new_df['low'] = df.resample(timeframe, on='time').low.last()
    new_df['close'] = df.resample(timeframe, on='time').close.last()
    new_df['volume'] = df.resample(timeframe, on='time').volume.sum()
    
    return new_df

def regular_hours(df):
    df = df[ ( (df.time.dt.hour*60 + df.time.dt.minute) >= 570) & ( (df.time.dt.hour*60 + df.time.dt.minute)<= 960)]
    return df

def find_beta(cl_df, stock_df):
    cl_returns = cl_df['close'].pct_change()[1:]
    stock_returns=stock_df['close'].pct_change()[1:]
    sns.regplot(cl_returns, stock_returns)
    plt.show()

    (beta, alpha) = stats.linregress(cl_returns, stock_returns)[0:2]
    return beta, alpha
    # beta = covariance[0,1]/covariance[1,1]
    # return beta



def main_data_runner(cl_perf_df, xom_df):

    length = len(cl_perf_df)
    # columns: time, futures_change, up/down, min or max, pct_change, time after cl_df move

    # initialize column arrays
    ftime, fchange, updown, initialvalue, minmax, pctchange, timeoccurred, timeafter = [[] for i in range(8)]


    for i in range(int(length)):

        futures_change = cl_perf_df.close_pct_change[i]
        ind = cl_perf_df.index[i]
        ftime.append(ind)
        fchange.append(futures_change)

        #print('futures: ', futures_change)
        # print(ind)


        index = xom_df.index.get_loc(ind)
        
        xom_df_filtered = xom_df[index:index+200]

        if futures_change < 0:
            # negative return
            updown.append('down')

            initial = xom_df_filtered['close'][0]
            high_low = min(xom_df_filtered['close'])
            pct_change = (initial/high_low -1) * 100

            time_occurred =  xom_df_filtered[xom_df_filtered['close']==high_low].index[0]
            timeoccurred.append(time_occurred)
            time_occurred = time_occurred.to_pydatetime()
            time_delta = (time_occurred-ind.to_pydatetime()).total_seconds()/3600 # hours

        if futures_change > 0:
            # positive return
            updown.append('up')

            initial = xom_df_filtered['close'][0]
            high_low = max(xom_df_filtered['close'])
            pct_change = (high_low/initial -1) * 100


            time_occurred =  xom_df_filtered[xom_df_filtered['close']==high_low].index[0]
            timeoccurred.append(time_occurred)
            time_occurred = time_occurred.to_pydatetime()
            time_delta = (time_occurred-ind.to_pydatetime()).total_seconds()/3600 # hours



        if pct_change == 0:
            # this basically says,
            # if pct_change == 0 from the two prior if-statements
            # then we LOST money, so the stop loss is -5%
            pct_change = -5
        
        timeafter.append(time_delta)
        initialvalue.append(initial)
        minmax.append(high_low)
        pctchange.append(pct_change)
        # print("Initial close: ", initial)
        # print('Max: ', high_low)
        # print('pct change: ', pct_change)
        # print('\n\n')



    
    dfDict = {
        'futures_time' : ftime,
        'futures_change' : fchange,
        'initial_value': initialvalue,
        'up_down': updown,
        'min_max' : minmax,
        'pct_change' : pctchange,
        'time_occurred': timeoccurred,
        'time_after' : timeafter
    }

    df = pd.DataFrame.from_dict(dfDict)
    df.to_csv(dir_path()+'/final.csv')
    return df






def main_data_evaluator(df):
    print("number of trades total: ", len(df))
    num_failed = df[df['pct_change'] == -5]
    print("number of failed trades: ", len(num_failed))
    chance_of_failure = (len(num_failed) / len(df)) * 100


    avg_hold = df['time_after'].mean() # in hours
    avg_profit = df['pct_change'].mean()
    print('chance of failure: ', round(chance_of_failure), '%')
    print('avg hold time in hours: ', round(avg_hold))
    print('avg pct move: ', round( avg_profit), '%')


def main_trader_single(df, xom_df):


    df = df.loc[100]
    start_date = df.futures_time


    index = xom_df.index.get_loc(start_date)
    xom = xom_df[index:].dropna()


    total_stop_loss = -.04 # 4%
    partial_stop_loss = -.01 #1%
    gain_stop = .02 #2%
    
    print('futures pct change: ', df['pct_change'])
    if df['pct_change'] > 0:
        print('Long...')

        initial_entry = xom.close[0]
        entry = initial_entry

        max_gain = 0

        for close in xom.close[1:]:

            cur_return = close/entry - 1
            total_return = close/initial_entry - 1
            # print('close: ', close, end='\r')
            # print('total_return: ', total_return*100, '%', end='\r')
            # print('max gain: ', max_gain, end='\r')

            ## if statement to check for TOTAL stop loss. i.e. if whole trade has lost 5%
            if total_return <= total_stop_loss:
                exit_reason = 'total_stop'

                break
            else:
                ## if statement to check to see if trade has gone up 
                entry=close
                if total_return >= gain_stop:

                    if total_return >= max_gain:
                        max_gain = total_return

                    else:
                        if total_return-max_gain <= partial_stop_loss:
                            exit_reason = 'partial_stop'
                            break
    

    if df['pct_change'] < 0:
        print('Short...')
 
        initial_entry = xom.close[0]
        entry = initial_entry
        
        max_gain = 0

        for close in xom.close[1:]:


            cur_return = entry/close - 1
            total_return = initial_entry/close - 1
            # print('close: ', close, end='\r')
            # print('total_return: ', total_return*100, '%', end='\r')
            # print('max gain: ', max_gain, end='\r')

            ## if statement to check for TOTAL stop loss. i.e. if whole trade has lost 5%
            if total_return <= total_stop_loss:

                exit_reason = 'total_stop'
                break
            else:
                ## if statement to check to see if trade has gone up 
                entry=close
                if total_return >= gain_stop:

                    if total_return >= max_gain:
                        max_gain = total_return
                        

                    else:
                        if total_return-max_gain <= partial_stop_loss:
                            exit_reason = 'partial_stop'

                            break

    start_date = start_date.to_pydatetime()
    end_date = xom[xom['close']==close].index[0].to_pydatetime()
    start_date_string = start_date.strftime(datetime_format())
    end_date_string = end_date.strftime(datetime_format())

    returnDict = {
        'total_return':total_return*100, 
        'max_return': max_gain*100,
        'entry':initial_entry,
        'entry_date': start_date_string,
        'exit': close,
        'exit_date': end_date_string,
        'exit_reason' : exit_reason,
        'holding_hours': (end_date-start_date).total_seconds()/(3600),
        'holding_days': (end_date-start_date).total_seconds()/(3600*24)
    }

    return returnDict







def main_trader_multiple_swing(og_df, xom_df, inputs):
    # pr = cProfile.Profile()
    # pr.enable()
    total_stop_loss = inputs[0]
    partial_stop_loss = inputs[1]
    gain_stop = inputs[2]

    # total_stop_loss = -.03 # 4%
    # partial_stop_loss = -.01 #1%
    # gain_stop = .015 #2%
        

    index_list = [i for i in range(len(og_df))]
    xom_df = xom_df.set_index('time')
    xom_df['time_series'] = xom_df.index

    # column initialization
    totalReturns, tradeTypes, maxGains, initalEntries, startDateStrings, closes, endDateStrings, exitReasons, holdingHours, holdingDays = [[] for i in range(10)] 

    for i in index_list:

        df = og_df.loc[i]
        start_date = df.futures_time



        

        index = xom_df.index.get_loc(start_date)
 
        xom = xom_df[index:].dropna()
        xom_close = xom.close[1:]



        #print('futures pct change: ', df['pct_change'])
        if df['pct_change'] > 0:
            #print('Long...')
            trade_type = 'long'

            initial_entry = xom_close[0]
            entry = initial_entry

            max_gain = 0

            for close in xom_close:

                cur_return = close/entry - 1
                total_return = close/initial_entry - 1
                # print('close: ', close, end='\r')
                # print('total_return: ', total_return*100, '%', end='\r')
                # print('max gain: ', max_gain, end='\r')

                ## if statement to check for TOTAL stop loss. i.e. if whole trade has lost 5%
                if total_return <= total_stop_loss:
                    exit_reason = 'total_stop'

                    break
                else:
                    ## if statement to check to see if trade has gone up 
                    entry=close
                    if total_return >= gain_stop:

                        if total_return >= max_gain:
                            max_gain = total_return

                        else:
                            if total_return-max_gain <= partial_stop_loss:
                                exit_reason = 'partial_stop'
                                break
        

        if df['pct_change'] < 0:
            #print('Short...')
            trade_type = 'short'
  
            initial_entry = xom_close[0]
            entry = initial_entry
            
            max_gain = 0

            for close in xom_close:


                cur_return = entry/close - 1
                total_return = initial_entry/close - 1
                # print('close: ', close, end='\r')
                # print('total_return: ', total_return*100, '%', end='\r')
                # print('max gain: ', max_gain, end='\r')

                ## if statement to check for TOTAL stop loss. i.e. if whole trade has lost 5%
                if total_return <= total_stop_loss:

                    exit_reason = 'total_stop'
                    break
                else:
                    ## if statement to check to see if trade has gone up 
                    entry=close
                    if total_return >= gain_stop:

                        if total_return >= max_gain:
                            max_gain = total_return
                            

                        else:
                            if total_return-max_gain <= partial_stop_loss:
                                exit_reason = 'partial_stop'

                                break

        start_date_string = start_date
        start_date = datetime.strptime(start_date, datetime_format())
        end_date_string = xom[xom['close']==close].index[0]
        # start_date_string = start_date.strftime(datetime_format())
        end_date= datetime.strptime(end_date_string, datetime_format()) #.strftime(datetime_format())

        totalReturns.append(total_return*100)
        tradeTypes.append(trade_type)
        maxGains.append(max_gain*100)
        initalEntries.append(initial_entry)
        startDateStrings.append(start_date_string)
        closes.append(close)
        endDateStrings.append(end_date_string)
        exitReasons.append(exit_reason)
        holdingHours.append((end_date-start_date).total_seconds()/(3600))
        holdingDays.append((end_date-start_date).total_seconds()/(3600*24))


    returnDict = {
        'trade_type': tradeTypes,
        'total_return': totalReturns, 
        'max_return': maxGains,
        'entry':initalEntries,
        'entry_date': startDateStrings,
        'exit': closes,
        'exit_date': endDateStrings,
        'exit_reason' : exitReasons,
        'holding_hours': holdingHours,
        'holding_days': holdingDays
    }
    df = pd.DataFrame.from_dict(returnDict)
    #pr.disable()    
    #ps = pstats.Stats(pr).sort_stats(SortKey.TIME)
    #ps.print_stats(25)
    return df



def trade_evaluator(df):
    avg_return = df.total_return.mean()
    total_return = df.total_return.sum()

    loss_trades = len(df[df['total_return'] < 0]) / len(df) * 100

    longs = len(df[df['trade_type']=='long'])
    shorts = len(df[df['trade_type']!='long'])
    
    avg_hold_hours = df.holding_hours.mean()
    avg_hold_days = df.holding_days.mean()
    print('percent trades with losses: ', round(loss_trades), '%')
    print('avg return: ', avg_return)
    print('total return: ', total_return)
    print('total trades: ', len(df))
    print('total long trades: ', longs)
    print('total short trades: ', shorts)
    print('hold hours avg: ', avg_hold_hours)
    print('hold days avg: ', avg_hold_days)

def NAV_growth(df):
    navgrowth = []
    initial = 1000
    current = initial
    for trade in df['total_return'].values:
        current = current*(1+trade/100)
      
        #print(trade)
        #print(current)
        navgrowth.append(current)

    end_nav = current
    print('end nav: ', end_nav)
    return navgrowth


def find_optimal_stops_swing(df, xom_df):
    path = pickle_path() +'/navgrowth_list_swing.pkl'
    try:
        with open(path, 'rb') as fil:
            navgrowth_list = pickle.load(fil)
    except:
        total_stops = [-i for i in np.arange(0.0001, .10, (.10-0.0001)/10)]
        partial_stops = [-i for i in np.arange(0.0001, .02, (.02-0.0001)/10)]
        stop_gains = [i for i in np.arange(0.0001, .09, (.09-0.0001)/10)]

        inputs_inputs = []
        for index1 in range(len(total_stops)):
            for index2 in range(len(partial_stops)):
                for index3 in range(len(stop_gains)):
                    inputs_inputs.append([total_stops[index1], partial_stops[index2], stop_gains[index3]])
        
        # for index in range(len(total_stops)):
        #     inputs_inputs.append([total_stops[index], partial_stops[index], stop_gains[index]])

        navgrowth_list = []
        for index, inputs in enumerate(inputs_inputs):
            df_trader  = main_trader_multiple_swing(df, xom_df, inputs)
            trade_evaluator(df_trader)
            print('\n\n')
            NAVgrowth = NAV_growth(df_trader)
            navgrowth_list.append(tuple((index, NAVgrowth[-1], inputs)))
        
        with open(path, 'wb') as fil:
            pickle.dump(navgrowth_list, fil)
    
    x = navgrowth_list
    max = 0
    for index, tup in enumerate(x):
        if x[index][1] > max:
            max = x[index][1]
            max_index = index
    
    winner = x[max_index]
    return navgrowth_list, winner


def find_optimal_stops_intraday(df, xom_df):
    path = pickle_path() +'/navgrowth_list_intraday.pkl'
    try:
        
        with open(path, 'rb') as fil:
            navgrowth_list = pickle.load(fil)

    except:
        total_stops = [-i for i in np.arange(0.0001, .10, (.10-0.0001)/10)]
        partial_stops = [-i for i in np.arange(0.0001, .02, (.02-0.0001)/10)]
        stop_gains = [i for i in np.arange(0.0001, .09, (.09-0.0001)/10)]

        inputs_inputs = []
        for index1 in range(len(total_stops)):
            for index2 in range(len(partial_stops)):
                for index3 in range(len(stop_gains)):
                    inputs_inputs.append([total_stops[index1], partial_stops[index2], stop_gains[index3]])
        
        # for index in range(len(total_stops)):
        #     inputs_inputs.append([total_stops[index], partial_stops[index], stop_gains[index]])

        navgrowth_list = []
        for index, inputs in enumerate(inputs_inputs):
            df_trader  = main_trader_multiple_intraday(df, xom_df, inputs)
            trade_evaluator(df_trader)
            print('\n\n')
            NAVgrowth = NAV_growth(df_trader)
            navgrowth_list.append(tuple((index, NAVgrowth[-1], inputs)))
       

        with open(path, 'wb') as fil:
            pickle.dump(navgrowth_list, fil)
    

    x = navgrowth_list
    max = 0
    for index, tup in enumerate(x):
        if x[index][1] > max:
            max = x[index][1]
            max_index = index
    winner = x[max_index]
    return navgrowth_list, winner


def main_trader_multiple_intraday(og_df, xom_df, inputs):
    # pr = cProfile.Profile()
    # pr.enable()
    total_stop_loss = inputs[0]
    partial_stop_loss = inputs[1]
    gain_stop = inputs[2]

    # total_stop_loss = -.03 # 4%
    # partial_stop_loss = -.01 #1%
    # gain_stop = .015 #2%
        

    index_list = [i for i in range(len(og_df))]
    xom_df = xom_df.set_index('time')
    xom_df['time_series'] = xom_df.index

    # column initialization
    totalReturns, tradeTypes, maxGains, initalEntries, startDateStrings, closes, endDateStrings, exitReasons, holdingHours, holdingDays = [[] for i in range(10)] 

    for i in index_list:

       
        try:

            df = og_df.loc[i]
            start_date = df.futures_time

            end_date = start_date[0:11] + '15:59:00'

            # dtime = datetime.strptime(end_date, datetime_format())

            start_index = xom_df.index.get_loc(start_date)
            end_index = xom_df.index.get_loc(end_date)

            
            xom = xom_df[start_index:end_index].dropna()

            xom_close = xom.close[1:]
            


            #print('futures pct change: ', df['pct_change'])
            if df['pct_change'] > 0:
                #print('Long...')
                trade_type = 'long'

                initial_entry = xom_close[0]
                entry = initial_entry

                max_gain = 0

                for index, close in enumerate(xom_close):

                    cur_return = close/entry - 1
                    total_return = close/initial_entry - 1

                    # print('close: ', close, end='\r')
                    # print('total_return: ', total_return*100, '%', end='\r')
                    # print('max gain: ', max_gain, end='\r')

                    # if statement to sell before close of market
                    if index == (len(xom_close) - 1):
                        exit_reason = 'close_of_market'

                        break


                    ## if statement to check for TOTAL stop loss. i.e. if whole trade has lost 5%
                    if total_return <= total_stop_loss:
                        exit_reason = 'total_stop'

                        break
                    else:
                        ## if statement to check to see if trade has gone up 
                        entry=close
                        if total_return >= gain_stop:

                            if total_return >= max_gain:
                                max_gain = total_return

                            else:
                                if total_return-max_gain <= partial_stop_loss:
                                    exit_reason = 'partial_stop'
                                    break
            

            if df['pct_change'] < 0:
                #print('Short...')
                trade_type = 'short'
    
                initial_entry = xom_close[0]
                entry = initial_entry
                
                max_gain = 0

                for close in xom_close:


                    cur_return = entry/close - 1
                    total_return = initial_entry/close - 1
                    # print('close: ', close, end='\r')
                    # print('total_return: ', total_return*100, '%', end='\r')
                    # print('max gain: ', max_gain, end='\r')

                    ## if statement to check for TOTAL stop loss. i.e. if whole trade has lost 5%
                    if total_return <= total_stop_loss:

                        exit_reason = 'total_stop'
                        break
                    else:
                        ## if statement to check to see if trade has gone up 
                        entry=close
                        if total_return >= gain_stop:

                            if total_return >= max_gain:
                                max_gain = total_return
                                

                            else:
                                if total_return-max_gain <= partial_stop_loss:
                                    exit_reason = 'partial_stop'

                                    break
 
            
            start_date_string = start_date
            start_date = datetime.strptime(start_date, datetime_format())
            end_date_string = xom[xom['close']==close].index[0]
            # start_date_string = start_date.strftime(datetime_format())
            end_date= datetime.strptime(end_date_string, datetime_format()) #.strftime(datetime_format())

            totalReturns.append(total_return*100)
            tradeTypes.append(trade_type)
            maxGains.append(max_gain*100)
            initalEntries.append(initial_entry)
            startDateStrings.append(start_date_string)
            closes.append(close)
            endDateStrings.append(end_date_string)
            exitReasons.append(exit_reason)
            holdingHours.append((end_date-start_date).total_seconds()/(3600))
            holdingDays.append((end_date-start_date).total_seconds()/(3600*24))

        except:
            print('date error')


    returnDict = {
        'trade_type': tradeTypes,
        'total_return': totalReturns, 
        'max_return': maxGains,
        'entry':initalEntries,
        'entry_date': startDateStrings,
        'exit': closes,
        'exit_date': endDateStrings,
        'exit_reason' : exitReasons,
        'holding_hours': holdingHours,
        'holding_days': holdingDays
    }
    df = pd.DataFrame.from_dict(returnDict)
    #pr.disable()    
    #ps = pstats.Stats(pr).sort_stats(SortKey.TIME)
    #ps.print_stats(25)
    return df
    
def runner2():
    cl_df = get_final_dataframe('cl')
    xom_df = get_final_dataframe('xom')

    cl_df = date_filter(cl_df, '2010-09-05')
    xom_df = date_filter(xom_df, '2010-09-05')

    tc = 1
    cl_df = resample(cl_df, '{}T'.format(tc))
    xom_df = resample(xom_df, '{}T'.format(tc))

    xom_df['time'] = xom_df.index
    cl_df['time'] = cl_df.index
    print(xom_df)
    # xom_df = regular_hours(xom_df)
    # cl_df = regular_hours(cl_df)


def match_dates(cl_df, stock_df):
    new_df = stock_df.join(cl_df, how='inner', lsuffix='_stock', rsuffix='_cl')
    new_df=new_df.dropna()
    
    cl_df = pd.DataFrame(new_df[['close_cl', 'volume_cl']])
    cl_df = cl_df.rename(columns={'close_cl':'close', 'volume_cl':'volume'})
    stock_df = pd.DataFrame(new_df[['open_stock', 'high_stock', 'low_stock', 'close_stock', 'volume_stock']])
    stock_df = stock_df.rename(columns={'open_stock': 'open', 'high_stock': 'high', 'low_stock':'low', 'close_stock':'close', 'volume_stock':'volume'})

    return cl_df, stock_df



def main_trader_multiple_swing_ATR(og_df, xom_df, inputs):
    # pr = cProfile.Profile()
    # pr.enable()

    partial_stop_loss = inputs[1]
    gain_stop = inputs[2]

    # total_stop_loss = -.03 # 4%
    # partial_stop_loss = -.01 #1%
    # gain_stop = .015 #2%
        

    index_list = [i for i in range(len(og_df))]
    xom_df = xom_df.set_index('time')
    xom_df['time_series'] = xom_df.index

    # column initialization
    totalReturns, tradeTypes, maxGains, initalEntries, startDateStrings, closes, endDateStrings, exitReasons, holdingHours, holdingDays = [[] for i in range(10)] 

    for i in index_list:

        df = og_df.loc[i]
        start_date = df.futures_time



        

        index = xom_df.index.get_loc(start_date)
 
        xom = xom_df[index:].dropna()
        xom_close = xom.close[1:]

        #print(xom)
        #print(xom.iloc[0].close) # gives close



        #print('futures pct change: ', df['pct_change'])
        if df['pct_change'] > 0:
            #print('Long...')
            trade_type = 'long'

            initial_entry = xom_close[0]
            entry = initial_entry

            max_gain = 0

            for index, close in enumerate(xom_close):

                cur_return = close/entry - 1
                total_return = close/initial_entry - 1


                if index == 0:
                    atr = 3*abs(xom.iloc[index].high - xom.iloc[index].close)
                    #print(atr)
                if index > 14:
                    atr_df= xom[(index-14):index]
                    atr = ATR(atr_df, 3)

                if index > 0 & index <= 14:
                    atr_df = xom[0:index]
                    atr = ATR(atr_df, 3)



                ## if statement to check for TOTAL stop loss. i.e. if whole trade has lost 5%
                

                if close <= (initial_entry-atr):
                    exit_reason = 'ATR_stop'

                    break
                else:
                    ## if statement to check to see if trade has gone up 
                    entry=close
                    if total_return >= gain_stop:

                        if total_return >= max_gain:
                            max_gain = total_return

                        else:
                            if total_return-max_gain <= partial_stop_loss:
                                exit_reason = 'partial_stop'
                                break
        

        if df['pct_change'] < 0:
            #print('Short...')
            trade_type = 'short'
  
            initial_entry = xom_close[0]
            entry = initial_entry
            
            max_gain = 0

            for index, close in enumerate(xom_close):


                cur_return = entry/close - 1
                total_return = initial_entry/close - 1

                if index == 0:
                    atr = 3*abs(xom.iloc[index].high - xom.iloc[index].close)
                    
                    
                if index > 14:
                    atr_df= xom[(index-14):index]
                    atr = ATR(atr_df, 3)
                
                if index > 0 & index <= 14:
                    atr_df = xom[0:index]
                    atr = ATR(atr_df, 3)

                ## if statement to check for TOTAL stop loss. i.e. if whole trade has lost 5%
                if close >= (initial_entry+atr):

                    exit_reason = 'ATR_stop'
                    break
                else:
                    ## if statement to check to see if trade has gone up 
                    entry=close
                    if total_return >= gain_stop:

                        if total_return >= max_gain:
                            max_gain = total_return
                            

                        else:
                            if total_return-max_gain <= partial_stop_loss:
                                exit_reason = 'partial_stop'

                                break

        start_date_string = start_date
        start_date = datetime.strptime(start_date, datetime_format())
        end_date_string = xom[xom['close']==close].index[0]
        # start_date_string = start_date.strftime(datetime_format())
        end_date= datetime.strptime(end_date_string, datetime_format()) #.strftime(datetime_format())

        totalReturns.append(total_return*100)
        tradeTypes.append(trade_type)
        maxGains.append(max_gain*100)
        initalEntries.append(initial_entry)
        startDateStrings.append(start_date_string)
        closes.append(close)
        endDateStrings.append(end_date_string)
        exitReasons.append(exit_reason)
        holdingHours.append((end_date-start_date).total_seconds()/(3600))
        holdingDays.append((end_date-start_date).total_seconds()/(3600*24))


    returnDict = {
        'trade_type': tradeTypes,
        'total_return': totalReturns, 
        'max_return': maxGains,
        'entry':initalEntries,
        'entry_date': startDateStrings,
        'exit': closes,
        'exit_date': endDateStrings,
        'exit_reason' : exitReasons,
        'holding_hours': holdingHours,
        'holding_days': holdingDays
    }
    df = pd.DataFrame.from_dict(returnDict)
    #pr.disable()    
    #ps = pstats.Stats(pr).sort_stats(SortKey.TIME)
    #ps.print_stats(25)
    return df


def ATR(df, factor):
# Max of:
# Method 1: Current High less the current Low
# Method 2: Current High less the previous Close (absolute value)
# Method 3: Current Low less the previous Close (absolute value)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    maxes = []
    index = 0

    while index < len(df)-1:
        method1 = abs(high[index+1] - low[index+1])

        method2 = abs(high[index+1] - close[index])

        method3 = abs(low[index+1] - close[index])
        
        atr_max = max([method1, method2, method3])
        maxes.append(atr_max)
        index+=1
    ATR = 1/len(df) * sum(maxes)

    #print(ATR*factor)
    return ATR*factor





def NAV_plotter(df_traded, NAVgrowth):
    df_traded['nav_growth'] = NAVgrowth
    df_traded['trade_number'] = df_traded.index
    fig = px.line(df_traded, x='trade_number', y="nav_growth")
    fig.show()

def STOPS_plotter(navgrowth_list):
    navs = []
    total_stops_list = []
    partial_stops_list = []
    stop_gains_list = []
    all_navs = [tup[1] for tup in navgrowth_list]
    min_nav = min(all_navs)
    for tup in navgrowth_list:
        nav = tup[1]
        nav_difference = nav-min_nav
        navs.append(nav_difference)
        total_stops_list.append(tup[2][0])
        partial_stops_list.append(tup[2][1])
        stop_gains_list.append(tup[2][2])

    scatter_dict = {
        'NAV':navs,
        'total_stops':total_stops_list,
        'partial_stops':partial_stops_list,
        'stop_gains':stop_gains_list
    }
    scatter_df = pd.DataFrame.from_dict(scatter_dict)
    scatter_df = scatter_df.sample(frac=1)
    fig = px.scatter_3d(scatter_df, x='total_stops', y='partial_stops', z='stop_gains', size='NAV')
    # fig.update_layout(scene_zaxis_type="log")
    fig.show()


def runner():
    cl_df = get_final_dataframe('cl')
    xom_df = get_final_dataframe('xom')

    cl_df = date_filter(cl_df, '2018-09-01')
    xom_df = date_filter(xom_df, '2018-09-01')

    tc = 1
    cl_df = resample(cl_df, '{}T'.format(tc))
    xom_df = resample(xom_df, '{}T'.format(tc))

    xom_df['time'] = xom_df.index
    cl_df['time'] = cl_df.index

    xom_df = regular_hours(xom_df)
    cl_df = regular_hours(cl_df)

    cl_df, xom_df = match_dates(cl_df, xom_df)

    xom_df.to_csv(dir_path() + '/xom_df.csv')

    #beta, alpha = find_beta(cl_df, xom_df)

    cl_perf_df, xom_perf_df = find_specific_performance(cl_df, xom_df, .005)

    df = main_data_runner(cl_perf_df, xom_df) # produces main data
    df.to_csv(dir_path()+'/main_data_runner.csv')

if __name__ == '__main__':

    pr = cProfile.Profile()
    pr.enable()

    runner()

    #runner2()
    xom_df = pd.read_csv(dir_path() + '/xom_df.csv')


    df = pd.read_csv(dir_path()+'/main_data_runner.csv')


    inputs = [0, 0, .02]
    df_traded = main_trader_multiple_swing_ATR(df, xom_df, inputs)
    NAVgrowth = NAV_growth(df_traded)



    # ATR(xom_df, 5)

    # main_data_evaluator(df)

    ### to plot line graph of NAV Growth! ####
    # inputs = [-.005, 0.0000, 0.0001]
    # df_traded = main_trader_multiple_swing(df, xom_df, inputs)
    # trade_evaluator(df_traded)
    # NAVgrowth = NAV_growth(df_traded)
    #NAV_plotter(df_traded, NAVgrowth)

    ##### to plot stop losses/gains vs. returns ####
    # navgrowth_list, winner = find_optimal_stops_swing(df, xom_df)
    # print(winner)
    # STOPS_plotter(navgrowth_list)


    pr.disable()    
    ps = pstats.Stats(pr).sort_stats(SortKey.TIME)



































    # to unpack and create dataframe:
    
    #optimal = find_optimal_stops_swing(df, xom_df)
    #print(optimal)
   
    #df.to_csv(dir_path()+'/main_trader_multiple.csv')

    #df = pd.read_csv(dir_path() + '/main_trader_multiple.csv')

    

    # NAVgrowth = NAV_growth(df)

    # df['nav_growth'] = NAVgrowth
    # df.to_csv(dir_path() + '/trader_withNAV_3percent.csv')

    # ps.print_stats(25)


    # total_stop_loss = -.03 # 4%
    # partial_stop_loss = -.01 #1%
    # gain_stop = .015 #2%


    # print(df)
    # print('average return: ', )
    # print('holding days mean: ', df.holding_days.mean())
    

    # print('\n\n')
    # print('entry: ', initial_entry)
    # print('exit: ', exit_)
    # print('total return from trade: ', total_return*100)

    # print(cl_df)
    # print(xom_df)
    #beta = find_beta(cl_df, xom_df)
    #print(beta)

    # xom_df, cl_df = find_specific_performance(cl_df, xom_df)

    # print(xom_final_df.head(25))
    # print(timeseries.head(25))

    # df = date_filter(cl_final_df, '2020-09-20')
    # print(df.head())
