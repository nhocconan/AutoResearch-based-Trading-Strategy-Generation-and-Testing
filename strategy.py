#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ChandelierExit_Trend_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA(10) for trend filter
    ema10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate Chandelier Exit on daily data
    # ATR(22)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # 22-period high and low for Chandelier Exit
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    lowest_low = pd.Series(low).rolling(window=22, min_periods=22).min().values
    
    # Chandelier Exit: Long exit = highest_high - 3*ATR, Short exit = lowest_low + 3*ATR
    chandelier_long_exit = highest_high - 3.0 * atr
    chandelier_short_exit = lowest_low + 3.0 * atr
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 22  # Need enough data for ATR and channels
    
    for i in range(start_idx, n):
        if (np.isnan(ema10_1w_aligned[i]) or
            np.isnan(chandelier_long_exit[i]) or
            np.isnan(chandelier_short_exit[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend_filter = ema10_1w_aligned[i]
        
        if position == 0:
            # Enter long: close above Chandelier long exit AND above weekly EMA trend
            if close[i] > chandelier_long_exit[i] and close[i] > trend_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: close below Chandelier short exit AND below weekly EMA trend
            elif close[i] < chandelier_short_exit[i] and close[i] < trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Chandelier long exit
            if close[i] < chandelier_long_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Chandelier short exit
            if close[i] > chandelier_short_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals