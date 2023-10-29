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
plot_all(pivots, df)

# pandas.set_option('display.max_rows',None)
# print(talib.MAX(time_candle['high'],5))
df = time_candle

# levels = []
# for i in range(2, df.shape[0] - 2):
#   if SupportResistance.is_support(df, i):
#      Low = df['low'][i]
#     if SupportResistance.is_far_from_level(Low, levels, df):
#      levels.append((i, Low))
# elif SupportResistance.is_resistance(df, i):
#   High = df['high'][i]
#  if SupportResistance.is_far_from_level(High, levels, df):
#   levels.append((i, High))


#  def plot_all(levels, df):
#    fig, ax = plt.subplots(figsize=(16, 9))
#    plt.plot(df['time'],df['close'])
#   for level in levels:
#      plt.hlines(level[1], xmin = df['time'][level[0]], xmax =
#     max(df['time']), colors='blue', linestyle='--')
# plt.show()

# plot_all(levels, df)

#   for level in levels:
#      df.loc[level[0],'Support&Resistance'] = level[1]
# pandas.set_option('display.max_rows', None)
# print(df)

from sklearn.cluster import AgglomerativeClustering


def calculate_support_resistance(df, rolling_wave_length, num_clusters):
    date = df.index
    # Reset index for merging
    df.reset_index(inplace=True)

    # Create min and max waves
    max_waves_temp = df.high.rolling(rolling_wave_length).max().rename('waves')

    min_waves_temp = df.low.rolling(rolling_wave_length).min().rename('waves')
    max_waves = pandas.concat([max_waves_temp, pandas.Series(numpy.zeros(len(max_waves_temp)) + 1)], axis=1)
    min_waves = pandas.concat([min_waves_temp, pandas.Series(numpy.zeros(len(min_waves_temp)) + -1)], axis=1)
    #  Remove dups
    max_waves.drop_duplicates('waves', inplace=True)
    min_waves.drop_duplicates('waves', inplace=True)
    #  Merge max and min waves
    waves = max_waves._append(min_waves).sort_index()
    waves = waves[waves[0] != waves[0].shift()].dropna()
    # Find Support/Resistance with clustering using the rolling stats
    # Create [x,y] array where y is always 1
    x = numpy.concatenate((waves.waves.values.reshape(-1, 1),
                           (numpy.zeros(len(waves)) + 1).reshape(-1, 1)), axis=1)

    # Initialize Agglomerative Clustering
    cluster = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
    cluster.fit_predict(x)
    waves['clusters'] = cluster.labels_
    # Get index of the max wave for each cluster
    waves2 = waves.loc[waves.groupby('clusters')['waves'].idxmax()]
    df.index = date
    waves2.waves.drop_duplicates(keep='first', inplace=True)
    return waves2.reset_index().waves


def plot_all(levels, df):
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(df['time'], df['close'])
    for level in levels:
        plt.hlines(level, xmin=min(df['time']), xmax=
        max(df['time']), colors='blue', linestyle='--')
    plt.show()


    ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
    ax1.plot(df['close'], linewidth=3, color='#ff9800', alpha=0.6)
    ax1.set_title('df CLOSING PRICE')
    ax2.plot(df['plus_di_2'], color='#26a69a', label='+ DI 14', linewidth=3, alpha=0.3)
    ax2.plot(df['minus_di_2'], color='#f44336', label='- DI 14', linewidth=3, alpha=0.3)
    ax2.plot(df['adx_2'], color='#2196f3', label='ADX 14', linewidth=3)
    ax2.axhline(25, color='grey', linewidth=2, linestyle='--')
    ax2.legend()
    ax2.set_title('df ADX 14')
    plt.show()

    """
      
       """