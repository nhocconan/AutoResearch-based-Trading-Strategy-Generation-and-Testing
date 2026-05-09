#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

name = "1d_WideChannel_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for trend and channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly Donchian(40) for channel (wider channel for fewer trades)
    high_20w = pd.Series(df_1w['high']).rolling(window=40, min_periods=40).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=40, min_periods=40).min().values
    
    # Align weekly indicators to daily
    ema34_1w_d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    high_20w_d = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_d = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Daily volume filter: volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_d[i]) or np.isnan(high_20w_d[i]) or 
            np.isnan(low_20w_d[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema34_1w_d[i]
        upper = high_20w_d[i]
        lower = low_20w_d[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Enter long: break above upper channel with volume and above trend
            if close[i] > upper and close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower channel with volume and below trend
            elif close[i] < lower and close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below midpoint of channel
            midpoint = (upper + lower) * 0.5
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above midpoint of channel
            midpoint = (upper + lower) * 0.5
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals