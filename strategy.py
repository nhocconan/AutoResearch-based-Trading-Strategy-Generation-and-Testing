#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyHighLow_Momentum_Volume"
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
    
    # Get weekly data for high/low reference
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly high and low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align weekly high/low to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Daily trend filter: EMA(50)
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(ema50[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_high = weekly_high_aligned[i]
        weekly_low = weekly_low_aligned[i]
        ema50_val = ema50[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: close above weekly high + above EMA50 + volume filter
            if close[i] > weekly_high and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: close below weekly low + below EMA50 + volume filter
            elif close[i] < weekly_low and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly low
            if close[i] < weekly_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly high
            if close[i] > weekly_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals