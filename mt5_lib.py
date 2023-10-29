import MetaTrader5
import pandas
import talib
import numpy
import datetime
from pandas import DataFrame
from sklearn import pipeline

import SupportResistance
import matplotlib.pyplot as plt
from talib import abstract
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler


def start_mt5(project_settings):
    #Set variables
    username = project_settings['mt5']['username']
    username = int(username)
    password = project_settings['mt5']['password']
    server = project_settings['mt5']['server']
    mt5_pathway = project_settings['mt5']['mt5_pathway']

    mt5_init = False
    try:
        mt5_init = MetaTrader5.initialize(
            login = username,
            password = password,
            server = server,
            path = mt5_pathway
        )
    except Exception as e:
        print(f"Error to initialize:{e}")
        #Return False
        mt5_init = False


    mt5_login = False
    if mt5_init:
        try:
            mt5_login = MetaTrader5.login(
                login=username,
                password=password,
                server=server,
                path=mt5_pathway)

        except Exception as e:
            print(f"Error to login{e}")
            mt5_login = False

    if mt5_login:
        return True
    return False

def initialize_symbol(symbol):
    all_symbols = MetaTrader5.symbols_get()
    symbol_names = []
    for sym in all_symbols:
        symbol_names.append(sym.name)

    if symbol in symbol_names:
        try:
            MetaTrader5.symbol_select(symbol, True)
            return True
        except Exception as e:
            print(f"Error enabling {symbol}. Error: {e}")
            return False
    else:
        print(f"Symbol {symbol} doesn't exist")
        return False

def get_candlesticks(symbol, timeframe, number_of_candles):
    if number_of_candles > 90000:
        raise ValueError("No more than 50000 candles can be retrieved")
    mt5_timeframe = set_query_timeframe(timeframe=timeframe)
    candles = MetaTrader5.copy_rates_from_pos(symbol, mt5_timeframe, 1, number_of_candles)
    dataframe = pandas.DataFrame(candles)
    return dataframe

"""
def prepare_candlesticks(dataframe):
    #dataframe['time'] = dataframe['time'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%H:%M'))
    lags = range(1,4)
    dataframelag = dataframe.assign(**{
        f'{col} (t-{lag})': dataframe[col].shift(lag)
        for lag in lags
        for col in dataframe
    })
    dataframelag.drop(dataframelag.head(4).index ,inplace=True)
    return dataframelag
"""
def prepare_candlesticks(df):
        lags = range(0, 10)
        lagged_columns = []
        for lag in lags:
            for col in df:
                new_col_name = f'{col} (t-{lag})'
                lagged_col = df[col].shift(lag)
                lagged_columns.append(lagged_col.rename(new_col_name))
        df = pandas.concat(lagged_columns, axis=1)
        return df

def prepare_candlesticks2(df):
    train, test = Split_Train_Test(df, 0.3)
    lags = range(0, 3)
    train_XX = train.loc[:, train.columns != 'signal']
    train_XXlag = train_XX.assign(**{
        f'{col} (t-{lag})': train_XX[col].shift(lag)
        for lag in lags
        for col in train_XX
    })
    Y_train = train["signal"]
    Y_train.drop(Y_train.head(5).index, inplace=True)  # drop last n rows
    train_XXlag.drop(train_XXlag.head(5).index, inplace=True)  # drop last n rows

    test_XX = test.loc[:, test.columns != 'signal']
    test_XXlag = test_XX.assign(**{
        f'{col} (t-{lag})': test_XX[col].shift(lag)
        for lag in lags
        for col in test_XX
    })

    Y_test = test["signal"]
    Y_test.drop(Y_test.head(5).index, inplace=True)  # drop last n rows
    test_XXlag.drop(test_XXlag.head(5).index, inplace=True)  # drop last n rows

    train_XX = train.loc[:, train.columns != 'signal']
    X_train = pandas.concat(
        [train_XX.shift(1), train_XX.shift(2), train_XX.shift(3), train_XX.shift(4), train_XX.shift(5)],
        axis=1).dropna()

    Y_train = train["signal"]
    Y_train.drop(Y_train.head(5).index, inplace=True)  # drop last n rows

    #####################
    ### test
    test_XX = test.loc[:, test.columns != 'signal']
    # X_test=pd.concat([test.shift(1)], axis=1).dropna()
    X_test = pandas.concat([test_XX.shift(1), test_XX.shift(2), test_XX.shift(3), test_XX.shift(4), test_XX.shift(5)],
                           axis=1).dropna()

    Y_test = test["signal"]
    Y_test.drop(Y_test.head(5).index, inplace=True)  # drop last n rows

    train_XX = train.loc[:, train.columns != 'signal']
    X_train = pandas.concat([train_XX.shift(1), train_XX.shift(2), train_XX.shift(3), train_XX.shift(4), train_XX.shift(5)],
                        axis=1).dropna()

    Y_train = train["signal"]
    Y_train.drop(Y_train.head(5).index, inplace=True)  # drop last n rows

    #####################
    ### test
    test_XX = test.loc[:, test.columns != 'signal']
    # X_test=pd.concat([test.shift(1)], axis=1).dropna()
    X_test = pandas.concat([test_XX.shift(1), test_XX.shift(2), test_XX.shift(3), test_XX.shift(4), test_XX.shift(5)],
                       axis=1).dropna()

    Y_test = test["signal"]
    Y_test.drop(Y_test.head(5).index, inplace=True)  # drop last n rows

    return X_train, Y_train, X_test, Y_test

def get_resistance_support(df):
    levels = []
    for i in range(2, df.shape[0] - 2):
       if SupportResistance.is_support(df, i):
            Low = df['low'][i]
            if SupportResistance.is_far_from_level(Low, levels, df):
                levels.append((i, Low))
       elif SupportResistance.is_resistance(df, i):
            High = df['high'][i]
            if SupportResistance.is_far_from_level(High, levels, df):
                levels.append((i, High))
    return levels

def get_resistance_support2(df):
    pivots = []
    max_list = []
    min_list = []
    for i in range(5, len(df) - 5):
        # taking a window of 9 candles
        high_range = df['high'][i - 5:i + 4]
        current_max = high_range.max()
        # if we find a new maximum value, empty the max_list
        if current_max not in max_list:
            max_list = []
        max_list.append(current_max)
        # if the maximum value remains the same after shifting 5 times
        if len(max_list) == 5 and SupportResistance.is_far_from_level(current_max, pivots, df):
            pivots.append((high_range.idxmax(), current_max))

        low_range = df['low'][i - 5:i + 5]
        current_min = low_range.min()
        if current_min not in min_list:
            min_list = []
        min_list.append(current_min)
        if len(min_list) == 5 and SupportResistance.is_far_from_level(current_min, pivots, df):
            pivots.append((low_range.idxmin(), current_min))

    return pivots

def plot_all_resistance_support(levels, df):
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(range(len(df)),df['close'])
    for level in levels:
        plt.hlines(level[1], xmin = level[0], xmax =
        max(range(len(df))), colors='blue', linestyle='--')
    plt.show()

def distance_to_close_supres(pivots,df):
        pivots = pandas.DataFrame(pivots)
        df['closest_pivot'] = df.apply(lambda row: min(pivots[1], key=lambda x: abs(x - row['high'])), axis=1)
        df['distancebetweenpivot'] = abs(df['closest_pivot']-df['high'])
        return df

def set_query_timeframe(timeframe):
    if timeframe == "M1":
        return MetaTrader5.TIMEFRAME_M1
    elif timeframe == "M2":
        return MetaTrader5.TIMEFRAME_M2
    elif timeframe == "M3":
        return MetaTrader5.TIMEFRAME_M3
    elif timeframe == "M4":
        return MetaTrader5.TIMEFRAME_M4
    elif timeframe == "M5":
        return MetaTrader5.TIMEFRAME_M5
    elif timeframe == "M6":
        return MetaTrader5.TIMEFRAME_M6
    elif timeframe == "M10":
        return MetaTrader5.TIMEFRAME_M10
    elif timeframe == "M12":
        return MetaTrader5.TIMEFRAME_M12
    elif timeframe == "M15":
        return MetaTrader5.TIMEFRAME_M15
    elif timeframe == "M20":
        return MetaTrader5.TIMEFRAME_M20
    elif timeframe == "M30":
        return MetaTrader5.TIMEFRAME_M30
    elif timeframe == "H1":
        return MetaTrader5.TIMEFRAME_H1
    elif timeframe == "H2":
        return MetaTrader5.TIMEFRAME_H2
    elif timeframe == "H3":
        return MetaTrader5.TIMEFRAME_H3
    elif timeframe == "H4":
        return MetaTrader5.TIMEFRAME_H4
    elif timeframe == "H6":
        return MetaTrader5.TIMEFRAME_H6
    elif timeframe == "H8":
        return MetaTrader5.TIMEFRAME_H8
    elif timeframe == "H12":
        return MetaTrader5.TIMEFRAME_H12
    elif timeframe == "D1":
        return MetaTrader5.TIMEFRAME_D1
    elif timeframe == "W1":
        return MetaTrader5.TIMEFRAME_W1
    elif timeframe == "MN1":
        return MetaTrader5.TIMEFRAME_MN1
    else:
        print(f"incorrect timeframe provided, {timeframe}")
        raise ValueError("Input incorrect timeframe")

'''
def ADX(df, window):
    indicator_adx = talib.ADX(high=df['high'], low=df['low'], close=df['close'], timeperiod=window)
    return indicator_adx

def ADX_PLUS(df, window):
    indicator_adx = talib.PLUS_DI(high=df['high'], low=df['low'], close=df['close'], timeperiod=window)
    return indicator_adx

def ADX_MINUS(df, window):
    indicator_adx = talib.MINUS_DI(high=df['high'], low=df['low'], close=df['close'], timeperiod=window)
    return indicator_adx
'''

def get_adx(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pandas.DataFrame(high - low)
    tr2 = pandas.DataFrame(abs(high - close.shift(1)))
    tr3 = pandas.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pandas.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(lookback).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1 / lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha=1 / lookback).mean()
    return plus_di, minus_di, adx_smooth

def ADX_data(df, adx_period = 14, plot=False):
    df['ADX'] = pandas.DataFrame(get_adx(df['high'], df['low'], df['close'], adx_period)[2]).rename(columns={0: 'ADX'})
    df['PLUS_DI'] = pandas.DataFrame(get_adx(df['high'], df['low'], df['close'], adx_period)[0]).rename(
        columns={0: 'PLUS_DI'})
    df['MINUS_DI'] = pandas.DataFrame(get_adx(df['high'], df['low'], df['close'], adx_period)[1]).rename(
        columns={0: 'MINUS_DI'})
    df['ADX_signal'] = pandas.DataFrame(implement_adx_strategy(df['close'], df['PLUS_DI'], df['MINUS_DI'], df['ADX'])).rename(columns={0: 'ADX_signal'})
    if plot:
        plot_adx(df)
    return df

def implement_adx_strategy(prices, pdi, ndi, adx):
    adx_signal = [0]
    signal = 0

    for i in range(1,len(prices)):
        if adx[i - 1] < 25 and adx[i] > 25 and pdi[i] > ndi[i]:
            if signal != 1:
                signal = 1
                adx_signal.append(signal)
            else:
                adx_signal.append(signal)
        elif adx[i - 1] < 25 and adx[i] > 25 and ndi[i] > pdi[i]:
            if signal != -1:
                signal = -1
                adx_signal.append(signal)
            else:
                adx_signal.append(signal)
        else:
            adx_signal.append(adx_signal[i-1])
    return adx_signal

def Supertrend(df, atr_period, multiplier):
    high = df['high']
    low = df['low']
    close = df['close']

    # calculate ATR
    price_diffs = [high - low,
                   high - close.shift(),
                   close.shift() - low]
    true_range = pandas.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()
    # df['atr'] = df['tr'].rolling(atr_period).mean()

    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)

    # initialize Supertrend column to True
    supertrend = [True] * len(df)

    for i in range(1, len(df.index)):
        curr, prev = i, i - 1

        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]

            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = numpy.nan
        else:
            final_lowerband[curr] = numpy.nan

    return pandas.DataFrame({
        'Supertrend': supertrend,
        'Final Lowerband': final_lowerband,
        'Final Upperband': final_upperband
    }, index=df.index)


def get_supertrend2(high, low, close, lookback, multiplier):
    # ATR

    tr1 = pandas.DataFrame(high - low)
    tr2 = pandas.DataFrame(abs(high - close.shift(1)))
    tr3 = pandas.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pandas.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(lookback).mean()

    # H/L AVG AND BASIC UPPER & LOWER BAND

    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()

    # FINAL UPPER BAND
    final_bands = pandas.DataFrame(columns=['upper', 'lower'])
    final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i - 1, 0]) | (close[i - 1] > final_bands.iloc[i - 1, 0]):
                final_bands.iloc[i, 0] = upper_band[i]
            else:
                final_bands.iloc[i, 0] = final_bands.iloc[i - 1, 0]

    # FINAL LOWER BAND

    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i - 1, 1]) | (close[i - 1] < final_bands.iloc[i - 1, 1]):
                final_bands.iloc[i, 1] = lower_band[i]
            else:
                final_bands.iloc[i, 1] = final_bands.iloc[i - 1, 1]

    # SUPERTREND

    supertrend = pandas.DataFrame(columns=[f'supertrend_{lookback}'])
    supertrend.iloc[:, 0] = [x for x in final_bands['upper'] - final_bands['upper']]

    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]

    # ST UPTREND/DOWNTREND

    upt = []
    dt = []
    """"
    print(close)
    close = close.iloc[len(close) - len(supertrend):]
    print(close)"""

    for i in range(len(supertrend)):
        if close[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(numpy.nan)
        elif close[i] < supertrend.iloc[i, 0]:
            upt.append(numpy.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(numpy.nan)
            dt.append(numpy.nan)

    st, upt, dt = pandas.DataFrame(supertrend.iloc[:, 0]), pandas.DataFrame(upt).rename(columns={0: 'upt'}), pandas.DataFrame(dt).rename(columns={0: 'dt'})
    st = st.rename(columns={0: 'supertrend'})
    upt.index, dt.index = supertrend.index, supertrend.index

    return st, upt, dt


def implement_st_strategy(prices, st):
    st_signal = [0]
    signal = 0

    for i in range(len(st)):
        if i ==0:
            continue
        if st[i - 1] > prices[i - 1] and st[i] < prices[i]:
            if signal != 1:
                signal = 1
                st_signal.append(signal)
            else:
                st_signal.append(signal)
        elif st[i - 1] < prices[i - 1] and st[i] > prices[i]:
            if signal != -1:
                signal = -1
                st_signal.append(signal)
            else:
                st_signal.append(signal)
        else:
            st_signal.append(st_signal[i - 1])

    return st_signal

def plot_ATR(df):
    plt.plot(df['close'], linewidth=2)
    plt.plot(df['upt'], color='green', linewidth=2, label='ST UPTREND')
    plt.plot(df['dt'], color='r', linewidth=2, label='ST DOWNTREND')
    plt.title('df ST TRADING SIGNALS')
    plt.legend(loc='upper left')
    plt.show()

def ATR(df, atr_period = 14, multiplier = 3, plot = False):
    ATR = get_supertrend2(df['high'], df['low'], df['close'], lookback=atr_period, multiplier=multiplier)
    df = pandas.concat([df, ATR[0],ATR[1],ATR[2]], axis=1)
    ST = pandas.DataFrame(implement_st_strategy(df['close'],df[f'supertrend_{atr_period}'])).rename(columns={0: 'supertrend_signal'})
    df = pandas.concat([df,ST], axis=1)
    if plot:
        plot_ATR(df)
    df = df.drop('upt', axis = 1)
    df = df.drop('dt', axis = 1)
    #df = df.drop('time',axis =1)
    return df
    
def plot_adx(df):
    ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
    ax1.plot(df['close'], linewidth=3, color='#ff9800', alpha=0.6)
    ax1.set_title('df CLOSING PRICE')
    ax2.plot(df['PLUS_DI'], color='#26a69a', label='+ DI 14', linewidth=3, alpha=0.3)
    ax2.plot(df['MINUS_DI'], color='#f44336', label='- DI 14', linewidth=3, alpha=0.3)
    ax2.plot(df['ADX'], color='#2196f3', label='ADX 14', linewidth=3)
    ax2.axhline(25, color='grey', linewidth=2, linestyle='--')
    ax2.legend()
    ax2.set_title('df ADX 14')
    plt.show()


def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    tr1 = pandas.DataFrame(high - low)
    tr2 = pandas.DataFrame(abs(high - close.shift()))
    tr3 = pandas.DataFrame(abs(low - close.shift()))
    frames = [tr1, tr2, tr3]
    tr = pandas.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(alpha=1 / atr_lookback).mean()

    kc_middle = close.ewm(kc_lookback).mean()
    kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
    kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr

    return kc_middle, kc_upper, kc_lower


def implement_kc_strategy(prices, kc_upper, kc_lower):
    kc_signal = [0]
    signal = 0

    for i in range(len(prices)):
        if i ==0:
            continue
        if prices[i-1] < kc_lower[i-1] and prices[i] > prices[i-1]:
            if signal != 1:
                signal = 1
                kc_signal.append(signal)
            else:
                kc_signal.append(signal)
        elif prices[i-1] > kc_upper[i-1] and prices[i] < prices[i-1]:
            if signal != -1:
                signal = -1
                kc_signal.append(signal)
            else:
                kc_signal.append(signal)
        else:
            kc_signal.append(kc_signal[i-1])

    return kc_signal

def kelner(df, kc_lookback=20, multiplier=2, atr_lookback=10, plot=False):
    df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], kc_lookback, multiplier, atr_lookback)
    df['kc_signal'] = implement_kc_strategy(df['close'], df['kc_upper'], df['kc_lower'])
    if plot:
        plot_kelner(df)
    return df

def plot_kelner(df):
    plt.plot(df['close'], linewidth=2, label='INTC')
    plt.plot(df['kc_upper'], linewidth=2, color='orange', linestyle='--', label='KC UPPER 20')
    plt.plot(df['kc_middle'], linewidth=1.5, color='grey', label='KC MIDDLE 20')
    plt.plot(df['kc_lower'], linewidth=2, color='orange', linestyle='--', label='KC LOWER 20')
    plt.legend(loc='lower right')
    plt.title('INTC KELTNER CHANNEL 20 TRADING SIGNALS')
    plt.show()

def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pandas.Series(up)
    down_series = pandas.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pandas.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)

    rsi_signal = []
    signal = 0
    for i in range(len(close)):
        if rsi[i] < 30:
            if signal != 1:
                signal = 1
                rsi_signal.append(signal)
            else:
                rsi_signal.append(signal)

        elif rsi[i] > 70:
            if signal != -1:
                signal = -1
                rsi_signal.append(signal)
            else:
                rsi_signal.append(signal)
        else:
            rsi_signal.append(0)

    return rsi_df, rsi_signal

def rsi_data(df,lookback = 14, plot=False):
    df['rsi'], df['rsi_signal'] = get_rsi(df['close'] , lookback)
    if plot:
        rsi_plot(df)
    return df


def rsi_plot(df):
    ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
    ax1.plot(df['close'])
    ax1.set_title('DF STOCK PRICE')
    ax2.plot(df['rsi'], color='orange', linewidth=1.5)
    ax2.axhline(30, color='grey', linestyle='--', linewidth=1.5)
    ax2.axhline(70, color='grey', linestyle='--', linewidth=1.5)
    ax2.set_title('DF RSI 14')
    plt.show()

def vwap(g):
   g['vwap'] = (g.tp * g.tick_volume).cumsum() / g.tick_volume.cumsum()
   return g

def VWAP_function(df, plot=False):
    copy_df = df.copy()
    copy_df['tp'] = (copy_df['low'] + copy_df['close'] + copy_df['high']) / 3
    copy_df['time'] =pandas.to_datetime(copy_df['time'], unit ='s')
    copy_df['day'] = copy_df['time'].dt.day
    copy_df = copy_df.groupby(pandas.Grouper(key ='day')).apply(lambda x: vwap(x))
    det = copy_df['vwap'].droplevel(level='day')
    df['vwap'] = det
    df['vwap_distance'] = abs(df['vwap']-df['close'])
    if plot:
        vwap_plot(df)
    return df


def vwap_plot(df):
     plt.plot(df['close'], linewidth=2, label='close')
     plt.plot(df['vwap'], linewidth=2, color='orange', linestyle='--', label='vwap')
     plt.legend(loc='lower right')
     plt.title('Close vwap CHANNEL')
     plt.show()

def detect_candlestick_patterns(df, pattern_list):
    #  Get the individual price columns
    open_prices = df['open']
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']

    #  Add a column for each candle pattern
    for candle_name in pattern_list:
        df[candle_name] = getattr(talib, candle_name)(open_prices, high_prices, low_prices, close_prices)
    #  Drop all columns with all zero values (which means: no pattern detected)
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def candle_type(df, plot=False):
    pattern_list = [method for method in dir(abstract) if method.startswith('CDL')]
    #  Detect candlestick patterns
    df = detect_candlestick_patterns(df, pattern_list)
    #  Create list of patterns detected during the period
    detected_pattern_list = []
    for column_name in df:
        if column_name.startswith('CDL'):
            detected_pattern_list.append(column_name)
        #  Create plot
    if plot:
        fig = plot_chart("XAUUSD", df, detected_pattern_list)
        fig.show()
    return df

def plot_chart(symbol, df, pattern_list):
    light_palette = {}
    light_palette["bg_color"] = "#ffffff"
    light_palette["plot_bg_color"] = "#ffffff"
    light_palette["grid_color"] = "#e6e6e6"
    light_palette["text_color"] = "#2e2e2e"
    light_palette["dark_candle"] = "#4d98c4"
    light_palette["light_candle"] = "#cccccc"
    light_palette["volume_color"] = "#f5f5f5"
    light_palette["border_color"] = "#2e2e2e"
    light_palette["color_1"] = "#5c285b"
    light_palette["color_2"] = "#802c62"
    light_palette["color_3"] = "#a33262"
    light_palette["color_4"] = "#c43d5c"
    light_palette["color_5"] = "#de4f51"
    light_palette["color_6"] = "#f26841"
    light_palette["color_7"] = "#fd862b"
    light_palette["color_8"] = "#ffa600"
    light_palette["color_9"] = "#3366d6"

    palette = light_palette

    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} Chart"],
                        specs=[[{"secondary_y": True}]],
                        vertical_spacing=0.04, shared_xaxes=True)

    #  Plot close price
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], close=df['close'], low=df['low'],
                                 high=df['high'], name='close'), row=1, col=1)
    #  Add candlestick pattern annotations
    periods_with_candles = []
    for pattern_name in pattern_list:
        for i, value in enumerate(df[pattern_name]):
            #  Skip this period in case it already has a pattern annotation
            if i in periods_with_candles:
                continue
                #  Skip periods where pattern is not identified
            if value == 0:
                continue
            x = i
            y = df['high'].iloc[i] + 8
            text_color = '#5c285b'
            if value > 0:
                text_color = 'LightSeaGreen'
                fig.add_annotation(x=x, y=y, textangle=-90, text=pattern_name, showarrow=False,
                                   font=dict(size=9, color=text_color))
                periods_with_candles.append(i)
    fig.update_layout(
        title={'text': '', 'x': 0.5},
        font=dict(family="Verdana", size=12, color=palette["text_color"]),
        autosize=True,
        width=1280, height=720,
        xaxis={"rangeslider": {"visible": False}},
        plot_bgcolor=palette["plot_bg_color"],
        paper_bgcolor=palette["bg_color"])
    fig.update_yaxes(visible=False, secondary_y=True)
    #  Change grid color
    fig.update_xaxes(showline=True, linewidth=1, linecolor=palette["grid_color"], gridcolor=palette["grid_color"])
    fig.update_yaxes(showline=True, linewidth=1, linecolor=palette["grid_color"], gridcolor=palette["grid_color"])
    #  Create output file
    file_name = f"{symbol}_candle_pattern_chart.png"
    return fig

def averages_sma(df, period=14):
    df['SMA']=df['close'].rolling(window=period).mean()
    df['corr']=df['close'].rolling(window=period).corr(df['SMA'])
    return df

def apply_technical(df):
    df = averages_sma(df)
    df.loc[df['corr']<-1,'corr'] = -1
    df.loc[df['corr']>1, 'corr'] = 1
    df = candle_type(df)
    df = pandas.DataFrame(df)
    pivots = get_resistance_support2(df)
    df = distance_to_close_supres(pivots,df)
    df = VWAP_function(df)
    df = rsi_data(df)
    df = kelner(df)
    df = ATR(df)
    df = ADX_data(df)
    #df['return'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['signal'] = 1  # Create a new 'signal' column and initialize it with 0

    df.loc[df['close'].shift(-1) > df['close'], 'signal'] = 1
    df.loc[df['close'].shift(-1) < df['close'], 'signal'] = -1

    copy_rate = df.copy()
    df = prepare_candlesticks(copy_rate)
    pandas.set_option('display.max_columns', None)
    #df['return (t-0)'] = df['return (t-0)'].fillna(df['return (t-0)'].mean())
    df = df.dropna()

    return df

def split_data(df):
    df_xtrain = df.drop('return', axis=1)
    df_ytrain = df['return']
    return df_xtrain,df_ytrain

def scaling_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

def Split_Train_Test(df, test_ratio):
    '''splits data into a training and testing set'''
    train_set_size = 1 - int(len(df) * test_ratio)
    train_set = df[:train_set_size]
    test_set = df[train_set_size:]
    return train_set, test_set




























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































