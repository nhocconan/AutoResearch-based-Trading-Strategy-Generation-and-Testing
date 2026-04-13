#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with daily volume confirmation
    # Long: Close > H3 (resistance 3) AND volume > 1.2x 20-period average
    # Short: Close < L3 (support 3) AND volume > 1.2x 20-period average
    # Exit: Close < H3 for longs OR Close > L3 for shorts
    # Using 6h timeframe for moderate trade frequency, Camarilla pivots from daily
    # for structure, volume confirmation to avoid fakeouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Resistance levels
    r1 = pp + (high_1d - low_1d) * 1.1 / 12
    r2 = pp + (high_1d - low_1d) * 1.1 / 6
    r3 = pp + (high_1d - low_1d) * 1.1 / 4
    r4 = pp + (high_1d - low_1d) * 1.1 / 2
    
    # Support levels
    s1 = pp - (high_1d - low_1d) * 1.1 / 12
    s2 = pp - (high_1d - low_1d) * 1.1 / 6
    s3 = pp - (high_1d - low_1d) * 1.1 / 4
    s4 = pp - (high_1d - low_1d) * 1.1 / 2
    
    # Align daily Camarilla levels to 6h
    h3_1d = align_htf_to_ltf(prices, df_1d, r3)
    l3_1d = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get daily volume for confirmation (>1.2x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d[i]) or np.isnan(l3_1d[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + volume confirmation
        long_entry = (close[i] > h3_1d[i]) and vol_confirm
        short_entry = (close[i] < l3_1d[i]) and vol_confirm
        
        # Exit logic: price retraces back inside H3/L3 levels
        long_exit = close[i] < h3_1d[i]
        short_exit = close[i] > l3_1d[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0