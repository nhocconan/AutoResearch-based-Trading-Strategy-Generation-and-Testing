#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_21ema_cross_volume_spike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(21) for trend
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate 1d average volume (20-period) for volume filter
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_spike = vol_1d_current > 1.5 * vol_avg_20_1d_aligned[i]  # 50% above average
        
        price = close[i]
        
        # Long when price crosses above 21-day EMA with volume spike
        long_cross = price > ema_21_1d_aligned[i] and close[i-1] <= ema_21_1d_aligned[i-1]
        long_signal = long_cross and vol_spike
        
        # Short when price crosses below 21-day EMA with volume spike
        short_cross = price < ema_21_1d_aligned[i] and close[i-1] >= ema_21_1d_aligned[i-1]
        short_signal = short_cross and vol_spike
        
        # Exit when price crosses back through 21-day EMA
        exit_long = price < ema_21_1d_aligned[i] and close[i-1] >= ema_21_1d_aligned[i-1]
        exit_short = price > ema_21_1d_aligned[i] and close[i-1] <= ema_21_1d_aligned[i-1]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals