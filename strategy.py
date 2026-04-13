#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 60-day (10 weeks) high/low for weekly pivot context
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    weekly_high = high_series.rolling(window=60, min_periods=60).max().values
    weekly_low = low_series.rolling(window=60, min_periods=60).min().values
    
    # 60-day average volume for spike detection
    vol_series = pd.Series(volume)
    avg_vol_60 = vol_series.rolling(window=60, min_periods=60).mean().values
    
    # Daily EMA200 for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = 60
    for i in range(start, n):
        if (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or 
            np.isnan(avg_vol_60[i]) or np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above weekly high + volume spike + above daily EMA200
            if (price > weekly_high[i] and vol > 2.0 * avg_vol_60[i] and price > ema200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly low + volume spike + below daily EMA200
            elif (price < weekly_low[i] and vol > 2.0 * avg_vol_60[i] and price < ema200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below weekly low OR below daily EMA200
            if price < weekly_low[i] or price < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above weekly high OR above daily EMA200
            if price > weekly_high[i] or price > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_WeeklyBreakout_Volume_EMA200"
timeframe = "6h"
leverage = 1.0