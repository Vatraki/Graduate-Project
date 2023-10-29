import pandas as pd
import numpy as np
import math
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

def is_support(df,i):
  cond1 = df['low'][i] < df['low'][i-1]
  cond2 = df['low'][i] < df['low'][i+1]
  cond3 = df['low'][i+1] < df['low'][i+2]
  cond4 = df['low'][i-1] < df['low'][i-2]
  return (cond1 and cond2 and cond3 and cond4)

def is_resistance(df,i):
  cond1 = df['high'][i] > df['high'][i-1]
  cond2 = df['high'][i] > df['high'][i+1]
  cond3 = df['high'][i+1] > df['high'][i+2]
  cond4 = df['high'][i-1] > df['high'][i-2]
  return (cond1 and cond2 and cond3 and cond4)


def is_far_from_level(value, levels, df):
  ave =  np.mean(df['high'] - df['low'])
  return np.sum([abs(value-level)<ave for _,level in levels])==0


