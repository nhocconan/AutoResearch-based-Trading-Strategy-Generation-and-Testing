#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52)
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    
    tenkan = (high_9 + low_9) / 2
    kijun = (high_26 + low_26) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high_52 + low_52) / 2
    
    # TK Cross signals
    tk_cross_long = (tenkan > kijun) & (tenkan[:-1] <= kijun[:-1])  # bullish cross
    tk_cross_short = (tenkan < kijun) & (tenkan[:-1] >= kijun[:-1])  # bearish cross
    
    # Cloud: green when senkou_a > senkou_b, red when senkou_a < senkou_b
    cloud_green = senkou_a > senkou_b
    cloud_red = senkou_a < senkou_b
    
    # Trend filter: price above/below cloud
    price_above_cloud = df_1d['close'].values > np.maximum(senkou_a, senkou_b)
    price_below_cloud = df_1d['close'].values < np.minimum(senkou_a, senkou_b)
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align all to 6h
    tk_long_6h = align_htf_to_ltf(prices, df_1d, tk_cross_long)
    tk_short_6h = align_htf_to_ltf(prices, df_1d, tk_cross_short)
    cloud_green_6h = align_htf_to_ltf(prices, df_1d, cloud_green)
    cloud_red_6h = align_htf_to_ltf(prices, df_1d, cloud_red)
    price_above_cloud_6h = align_htf_to_ltf(prices, df_1d, price_above_cloud)
    price_below_cloud_6h = align_htf_to_ltf(prices, df_1d, price_below_cloud)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 52  # Need enough data for Ichimoku
    
    for i in range(start_idx, n):
        if (np.isnan(tk_long_6h[i]) or np.isnan(tk_short_6h[i]) or
            np.isnan(cloud_green_6h[i]) or np.isnan(cloud_red_6h[i]) or
            np.isnan(price_above_cloud_6h[i]) or np.isnan(price_below_cloud_6h[i]) or
            np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tk_long = tk_long_6h[i]
        tk_short = tk_short_6h[i]
        cloud_green = cloud_green_6h[i]
        cloud_red = cloud_red_6h[i]
        price_above = price_above_cloud_6h[i]
        price_below = price_below_cloud_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: bullish TK cross + price above cloud + green cloud + volume
            if tk_long and price_above and cloud_green and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish TK cross + price below cloud + red cloud + volume
            elif tk_short and price_below and cloud_red and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish TK cross or price below cloud
            if tk_short or price_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross or price above cloud
            if tk_long or price_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals