import json
import os
import pandas
import time
import talib as ta
import numpy
import mt5_lib
import matplotlib.pyplot as plt
import SupportResistance
import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from talib import abstract
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

setting_filepath = "settings.json"

def get_project_settings(import_filepath):
    if os.path.exists(import_filepath):
        f = open(import_filepath, "r")
        project_settings = json.load(f)
        f.close()
        return project_settings

    else:
        raise ImportError("Settings.json does not exist in location")


def start_up(project_settings):
    startup = mt5_lib.start_mt5(project_settings=project_settings)
    symbols = project_settings['mt5']['symbols']
    if startup:
        print("Metatrader is connected successfully")
        """for symbol in symbols:"""
        init_symbol = mt5_lib.initialize_symbol(symbols)
        if init_symbol is True:
            print(f"Symbol {symbols} initialized")
        else:
            raise Exception(f"{symbols} not initialized")
        """return True"""
    return False


if __name__ == '__main__':
    print("Let's build a bot")
    project_settings = get_project_settings(import_filepath=setting_filepath)
    startup = start_up(project_settings=project_settings)

    timeframe = project_settings['mt5']['timeframe']
    symbol = project_settings['mt5']['symbols']

    df = mt5_lib.get_candlesticks(
        symbol=symbol,
        timeframe=timeframe,
        number_of_candles=500)

    df = mt5_lib.apply_technical(df)
    #df = df[df['distancebetweenpivot (t-0)'] < 0.5]
    train_set, test_set = mt5_lib.Split_Train_Test(df, 0.2)
    #train_set = train_set[train_set['distancebetweenpivot (t-0)'] < 2]
    #test_set = test_set[test_set['distancebetweenpivot (t-0)'] < 2]
    x = train_set.drop(['signal (t-0)'], axis=1)
    y = test_set['signal (t-0)']
    pandas.set_option('display.max_columns', None)

    """
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)  # Adjust test_size as per your requirement
    """
    X_train = train_set.drop(['signal (t-0)'], axis=1)
    X_test = test_set.drop(['signal (t-0)'], axis=1)
    copy_test = X_test
    y_train = train_set['signal (t-0)']
    y_test = test_set['signal (t-0)']

    #scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.fit_transform(X_test)

    # Creating and training the model (using Random Forest as an example)
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    model = LinearSVC()
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    y_pred_mapped = numpy.where(y_pred > 0, 1, -1)

    pattern_list = [method for method in dir(abstract) if method.startswith('CDL')]

    for pred, actual in zip(y_pred, y_test):
        print("Predicted:", pred, "Actual:", actual)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy is {accuracy}")




"""
    df = df.rename(columns={'time (t-0)': 'Time', 'high (t-0)': 'High', 'open (t-0)': 'Open', 'low (t-0)': 'Low', 'close (t-0)': 'Close', 'tick_volume (t-0)': 'Volume', 'signal (t-0)': 'signal' })

    df['Time'] = pandas.to_datetime(df['Time'], unit='s')
    df = df.set_index('Time')

    def SIGNAL():
        return df.signal

    from backtesting import Strategy
    class MyCandlesStrat(Strategy):
        def init(self):
            super().init()
            self.signal1 = self.I(SIGNAL)

        def next(self):
            super().next()
            if self.signal1 == 1:
                sl1 = self.data.Close[-1] - 500
                tp1 = self.data.Close[-1] + 20
                self.buy(sl=sl1, tp=tp1)
            elif self.signal1 == -1:
                sl1 = self.data.Close[-1] + 500
                tp1 = self.data.Close[-1] - 20
                self.sell(sl=sl1, tp=tp1)


    from backtesting import Backtest

    bt = Backtest(df, MyCandlesStrat, cash=10_000, commission=.002)
    stat = bt.run()
    print(stat)
    bt.plot()
"""
