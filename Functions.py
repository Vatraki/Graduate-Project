import json
import os
import pandas
import time
import talib as ta
import numpy
import mt5_lib
import matplotlib.pyplot as plt
import SupportResistance
#from mplfinance.original_flavor import candlestick_ohlc

import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import talib as ta
from talib import abstract
import os


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


def detect_candlestick_patterns(df, pattern_list):
    #  Get the individual price columns
    open_prices = df['open']
    high_prices = df['high']
    low_prices = df['low']
    close_prices = df['close']

    #  Add a column for each candle pattern
    for candle_name in pattern_list:
        df[candle_name] = getattr(ta, candle_name)(open_prices, high_prices, low_prices, close_prices)
    #  Drop all columns with all zero values (which means: no pattern detected)
    df = df.loc[:, (df != 0).any(axis=0)]
    return df


pattern_list = [method for method in dir(abstract) if method.startswith('CDL')]
print(pattern_list)
#  Detect candlestick patterns
df = detect_candlestick_patterns(df, pattern_list)
#  Create list of patterns detected during the period
detected_pattern_list = []

for column_name in df:
    if column_name.startswith('CDL'):
        detected_pattern_list.append(column_name)
    #  Create plot

fig = plot_chart("XAUUSD", df, detected_pattern_list)
fig.show()

df = pandas.DataFrame(df)

df['adx1'] = mt5_lib.ADX(df, 14)

df['plus_di'] = pandas.DataFrame(mt5_lib.get_adx(df['high'], df['low'], df['close'], 14)[0]).rename(
    columns={0: 'plus_di'})
df['minus_di'] = pandas.DataFrame(mt5_lib.get_adx(df['high'], df['low'], df['close'], 14)[1]).rename(
    columns={0: 'minus_di'})
df['adx'] = pandas.DataFrame(mt5_lib.get_adx(df['high'], df['low'], df['close'], 14)[2]).rename(columns={0: 'adx'})


