#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d True Range and ATR (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Calculate ATR (14-period)
    atr = np.zeros(len(high_1d))
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    else:
        for i in range(len(tr)):
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
    
    # === 1d ATR Trailing Stop Logic ===
    # Long stop: highest high since entry minus ATR * multiplier
    # Short stop: lowest low since entry plus ATR * multiplier
    atr_mult = 3.0
    
    # Track highest high and lowest low since entry
    highest_since_entry = np.zeros(len(high_1d))
    lowest_since_entry = np.zeros(len(high_1d))
    
    highest_since_entry[0] = high_1d[0]
    lowest_since_entry[0] = low_1d[0]
    for i in range(1, len(high_1d)):
        highest_since_entry[i] = max(highest_since_entry[i-1], high_1d[i])
        lowest_since_entry[i] = min(lowest_since_entry[i-1], low_1d[i])
    
    # Calculate trailing stops
    long_stop = highest_since_entry - atr * atr_mult
    short_stop = lowest_since_entry + atr * atr_mult
    
    # === Align indicators to 6h timeframe ===
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    long_stop_aligned = align_htf_to_ltf(prices, df_1d, long_stop)
    short_stop_aligned = align_htf_to_ltf(prices, df_1d, short_stop)
    
    # === 6h Volume Spike Detection ===
    # Volume > 2.0 x 20-period average volume
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[0:i+1]) if i > 0 else volume[0]
    
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(long_stop_aligned[i]) or 
            np.isnan(short_stop_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above long_stop AND volume spike
            if close[i] > long_stop_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below short_stop AND volume spike
            elif close[i] < short_stop_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below long_stop
            if close[i] < long_stop_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above short_stop
            if close[i] > short_stop_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ATR_VolumeSpike_Breakout_v1"
timeframe = "6h"
leverage = 1.0