#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels: R1 = close + 1.1 * range / 12, S1 = close - 1.1 * range / 12
    r1_level = close_1d + (1.1 * daily_range / 12)
    s1_level = close_1d - (1.1 * daily_range / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume filter: current 4h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: close above R1 + above EMA34 trend + volume filter
            if close[i] > r1 and close[i] > ema34_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: close below S1 + below EMA34 trend + volume filter
            elif close[i] < s1 and close[i] < ema34_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA34 trend
            if close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above EMA34 trend
            if close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals