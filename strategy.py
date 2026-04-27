#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: require volume > 1.5x 20-period average (more selective than previous)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (active trading hours) - reduced from previous version
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 34  # need 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > EMA34 (bullish bias) + volume + session
            if (close[i] > ema34_1d_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < EMA34 (bearish bias) + volume + session
            elif (close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA34_Bias_Volume_Session"
timeframe = "12h"
leverage = 1.0