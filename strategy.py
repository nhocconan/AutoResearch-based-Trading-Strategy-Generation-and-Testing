#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_Breakout_Trend_21"
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
    
    # Get 1d data for trend filter (EMA21)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d EMA(21) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema21_1d = close_1d.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Calculate Donchian(20) channels from 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donch = high_series.rolling(window=20, min_periods=20).max().values
    lower_donch = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21_1d_aligned[i]) or np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and above 1d EMA21
            if close[i] > upper_donch[i] and vol_ok and close[i] > ema21_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume and below 1d EMA21
            elif close[i] < lower_donch[i] and vol_ok and close[i] < ema21_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price closes below lower Donchian
            if close[i] < lower_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above upper Donchian
            if close[i] > upper_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals