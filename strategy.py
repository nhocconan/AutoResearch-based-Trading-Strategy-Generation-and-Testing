#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate weekly high/low for trend filter
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_w, weekly_low)
    
    # Calculate daily ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    atr_sum = 0.0
    for i in range(n):
        atr_sum += tr[i]
        if i >= 14:
            atr_sum -= tr[i - 14]
        if i >= 13:
            atr[i] = atr_sum / 14
    
    # Calculate daily volume moving average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 19:
            vol_sum -= volume[i - 20]
        if i >= 18:
            vol_ma[i] = vol_sum / 20
    
    # Calculate daily close moving average for trend
    close_ma = np.full(n, np.nan)
    close_sum = 0.0
    for i in range(n):
        close_sum += close[i]
        if i >= 49:
            close_sum -= close[i - 50]
        if i >= 48:
            close_ma[i] = close_sum / 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(close_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: close below weekly low or ATR-based stop
            if close[i] < weekly_low_aligned[i] or close[i] < close[i-1] - 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above weekly high or ATR-based stop
            if close[i] > weekly_high_aligned[i] or close[i] > close[i-1] + 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above weekly high, above MA, and volume surge
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > close_ma[i] and 
                volume[i] > vol_ma[i] * 2.0):
                position = 1
                signals[i] = 0.25
            # Enter short: price below weekly low, below MA, and volume surge
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < close_ma[i] and 
                  volume[i] > vol_ma[i] * 2.0):
                position = -1
                signals[i] = -0.25
    
    return signals