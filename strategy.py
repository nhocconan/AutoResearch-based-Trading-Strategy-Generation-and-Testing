#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Resistance levels (tighter levels for fewer trades)
    h1_12h = close_12h + 1.1 * range_12h / 6
    h2_12h = close_12h + 1.1 * range_12h / 4
    h3_12h = close_12h + 1.1 * range_12h / 2
    
    # Support levels
    l1_12h = close_12h - 1.1 * range_12h / 6
    l2_12h = close_12h - 1.1 * range_12h / 4
    l3_12h = close_12h - 1.1 * range_12h / 2
    
    # Align 12h levels to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    h1_aligned = align_htf_to_ltf(prices, df_12h, h1_12h)
    h2_aligned = align_htf_to_ltf(prices, df_12h, h2_12h)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l1_aligned = align_htf_to_ltf(prices, df_12h, l1_12h)
    l2_aligned = align_htf_to_ltf(prices, df_12h, l2_12h)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    
    # 12h volume confirmation: current volume > 24-period average
    volume_12h = df_12h['volume'].values
    vol_avg_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    vol_avg_24_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_24)
    
    # 4h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 24 to ensure sufficient data
    for i in range(24, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(vol_avg_24_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 12h volume (aligned)
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        vol_confirm = vol_12h_current > vol_avg_24_aligned[i]
        
        # Volatility filter: only trade when ATR > 50-period average
        atr_avg_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean()[i]
        vol_filter = atr[i] > atr_avg_50
        
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