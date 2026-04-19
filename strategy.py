#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Pivot_Breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for multi-timeframe analysis
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h ATR for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr_4h = np.maximum(high_4h - low_4h, np.absolute(high_4h - np.roll(close_4h, 1)), np.absolute(low_4h - np.roll(close_4h, 1)))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 1d pivot levels (daily R1/S1) for structure
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1h volume filter: current volume > 1.5x 24-period average (24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        if np.isnan(atr_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or \
           np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above daily R1 with volume
            if price > r1_1d_aligned[i] and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: break below daily S1 with volume
            elif price < s1_1d_aligned[i] and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price returns to daily pivot or below
            if price < pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price returns to daily pivot or above
            if price > pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals