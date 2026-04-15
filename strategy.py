#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d high/low for range calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range (high - low)
    daily_range = high_1d - low_1d
    # Average daily range over last 20 days
    avg_range_20 = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    avg_range_20_aligned = align_htf_to_ltf(prices, df_1d, avg_range_20)
    
    # Current day's range
    current_range = high_1d - low_1d
    current_range_aligned = align_htf_to_ltf(prices, df_1d, current_range)
    
    # 4h EMA(20) for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current > 1.5x median of last 30 bars
    vol_median = pd.Series(volume).rolling(window=30, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(avg_range_20_aligned[i]) or 
            np.isnan(current_range_aligned[i]) or
            np.isnan(ema_20[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Range filter: trade only when current range < 60% of average range (low volatility)
        range_filter = current_range_aligned[i] < 0.6 * avg_range_20_aligned[i]
        
        # Long: price near daily low + volume + range filter + above EMA20
        if (close[i] <= low_1d[i] * 1.001 and  # within 0.1% of daily low
            volume[i] > vol_threshold[i] and 
            range_filter and
            close[i] > ema_20[i]):
            signals[i] = 0.25
        
        # Short: price near daily high + volume + range filter + below EMA20
        elif (close[i] >= high_1d[i] * 0.999 and  # within 0.1% of daily high
              volume[i] > vol_threshold[i] and 
              range_filter and
              close[i] < ema_20[i]):
            signals[i] = -0.25
        
        # Exit: price moves back toward middle of daily range
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] > (low_1d[i] + high_1d[i]) * 0.5) or
               (signals[i-1] == -0.25 and close[i] < (low_1d[i] + high_1d[i]) * 0.5))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyRangeBreakout_VolumeFilter"
timeframe = "4h"
leverage = 1.0