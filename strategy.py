#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        close_val = prices['close'].iloc[i]
        vol_val = df_1d['volume'].iloc[i] if i < len(df_1d) else volume_1d[-1]  # fallback for safety
        dh = donch_high_aligned[i]
        dl = donch_low_aligned[i]
        vavg = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(dh) or np.isnan(dl) or np.isnan(vavg):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high + volume > average
            if close_val > dh and vol_val > vavg:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low + volume > average
            elif close_val < dl and vol_val > vavg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 12h Donchian low
            if close_val < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h Donchian high
            if close_val > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hDonchian_Volume"
timeframe = "4h"
leverage = 1.0