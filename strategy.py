#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 24:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    h1_1d = close_1d + 1.1 * range_1d / 6
    h2_1d = close_1d + 1.1 * range_1d / 4
    h3_1d = close_1d + 1.1 * range_1d / 2
    
    # Support levels
    l1_1d = close_1d - 1.1 * range_1d / 6
    l2_1d = close_1d - 1.1 * range_1d / 4
    l3_1d = close_1d - 1.1 * range_1d / 2
    
    # Align 1d levels to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1_1d)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1_1d)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # 1d volume confirmation: current volume > 10-period average
    volume_1d = df_1d['volume'].values
    vol_avg_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_10_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10)
    
    # 4h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 10 to ensure sufficient data
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(vol_avg_10_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_10_aligned[i]
        
        # Volatility filter: only trade when ATR > 20-period average
        atr_avg_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean()[i]
        vol_filter = atr[i] > atr_avg_20
        
        # Price levels for current bar
        h1 = h1_aligned[i]
        h2 = h2_aligned[i]
        h3 = h3_aligned[i]
        l1 = l1_aligned[i]
        l2 = l2_aligned[i]
        l3 = l3_aligned[i]
        pivot = pivot_aligned[i]
        
        # Long conditions: bounce from support levels with volume and volatility
        long_signal = vol_confirm and vol_filter and (
            (close[i] > l1 and low[i] <= l1) or  # bounce from L1
            (close[i] > l2 and low[i] <= l2) or  # bounce from L2
            (close[i] > l3 and low[i] <= l3)     # bounce from L3
        )
        
        # Short conditions: rejection from resistance levels with volume and volatility
        short_signal = vol_confirm and vol_filter and (
            (close[i] < h1 and high[i] >= h1) or  # rejection from H1
            (close[i] < h2 and high[i] >= h2) or  # rejection from H2
            (close[i] < h3 and high[i] >= h3)     # rejection from H3
        )
        
        # Exit conditions: price moves to opposite side of pivot
        long_exit = close[i] < pivot
        short_exit = close[i] > pivot
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals