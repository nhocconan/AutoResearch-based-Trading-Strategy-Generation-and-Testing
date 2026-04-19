#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Pivot_R1S1_Breakout_VolumeATR_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for pivot calculation (once before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h high, low, close for pivot calculation
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1d high, low, close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h pivot points: P = (H+L+C)/3
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_4h = 2 * pivot_4h - low_4h
    s1_4h = 2 * pivot_4h - high_4h
    
    # Calculate 1d pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align 4h pivot levels to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Align 1d pivot levels to 1h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h ATR for volatility filter (14-period)
    tr4h = np.maximum(high_4h[1:] - low_4h[1:], np.absolute(high_4h[1:] - close_4h[:-1]))
    tr4h = np.maximum(tr4h, np.absolute(low_4h[1:] - close_4h[:-1]))
    tr4h = np.concatenate([[np.nan], tr4h])
    atr_14_4h = pd.Series(tr4h).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # 1h volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter (08-20 UTC)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot_4h_val = pivot_4h_aligned[i]
        r1_4h_val = r1_4h_aligned[i]
        s1_4h_val = s1_4h_aligned[i]
        pivot_1d_val = pivot_1d_aligned[i]
        r1_1d_val = r1_1d_aligned[i]
        s1_1d_val = s1_1d_aligned[i]
        atr = atr_14_4h_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Require both 4h and 1d levels to agree for stronger signal
        # Long: price above both 4h R1 and 1d R1 with volume
        # Short: price below both 4h S1 and 1d S1 with volume
        if position == 0:
            long_condition = (price > r1_4h_val and price > r1_1d_val and volume_confirmed)
            short_condition = (price < s1_4h_val and price < s1_1d_val and volume_confirmed)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price below either 4h pivot or 1d pivot, or ATR-based stop
            if (price < pivot_4h_val or price < pivot_1d_val or 
                price < close[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price above either 4h pivot or 1d pivot, or ATR-based stop
            if (price > pivot_4h_val or price > pivot_1d_val or 
                price > close[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals