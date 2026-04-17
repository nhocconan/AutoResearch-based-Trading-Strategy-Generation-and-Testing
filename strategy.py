#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Supertrend(10,3) trend filter + 6h Camarilla pivot breakout + volume confirmation.
Long when price breaks above Camarilla R3 level with 12h Supertrend uptrend and volume > 1.3x 20-period volume average.
Short when price breaks below Camarilla S3 level with 12h Supertrend downtrend and volume > 1.3x 20-period volume average.
Uses 12h timeframe for trend filter to avoid whipsaw, and 6h for precise entry/exit. Designed to work in both bull and bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10)
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = pd.Series(high_vals).shift(1).values
        tr2 = pd.Series(low_vals).shift(1).values
        tr0 = high_vals - low_vals
        tr1 = np.abs(high_vals - tr1)
        tr2 = np.abs(low_vals - tr2)
        tr = np.maximum(tr0, np.maximum(tr1, tr2))
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        # First values: use expanding mean
        tr_expanding = pd.Series(tr).expanding(min_periods=1).mean().values
        atr_vals = np.where(np.arange(len(tr)) < window, tr_expanding, atr_vals)
        return atr_vals
    
    atr_10_12h = atr(high_12h, low_12h, close_12h, 10)
    
    # Calculate 12h Supertrend(10,3)
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + 3 * atr_10_12h
    lower_band_12h = hl2_12h - 3 * atr_10_12h
    
    supertrend_12h = np.full(len(close_12h), np.nan)
    direction_12h = np.full(len(close_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(close_12h)):
        if i == 0:
            supertrend_12h[i] = hl2_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i] > supertrend_12h[i-1]:
                direction_12h[i] = 1
            elif close_12h[i] < supertrend_12h[i-1]:
                direction_12h[i] = -1
            else:
                direction_12h[i] = direction_12h[i-1]
            
            if direction_12h[i] == 1:
                supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
            else:
                supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
    
    # Calculate 6h Camarilla levels from previous 6h bar
    def camarilla_levels(high_val, low_val, close_val):
        range_val = high_val - low_val
        if range_val == 0:
            return close_val, close_val, close_val, close_val, close_val, close_val, close_val, close_val
        r4 = close_val + range_val * 1.1 / 2
        r3 = close_val + range_val * 1.1 / 4
        r2 = close_val + range_val * 1.1 / 6
        r1 = close_val + range_val * 1.1 / 12
        s1 = close_val - range_val * 1.1 / 12
        s2 = close_val - range_val * 1.1 / 6
        s3 = close_val - range_val * 1.1 / 4
        s4 = close_val - range_val * 1.1 / 2
        return r1, r2, r3, r4, s1, s2, s3, s4
    
    # Use previous bar's OHLC for Camarilla (standard practice)
    r1_6h = np.full(n, np.nan)
    r2_6h = np.full(n, np.nan)
    r3_6h = np.full(n, np.nan)
    r4_6h = np.full(n, np.nan)
    s1_6h = np.full(n, np.nan)
    s2_6h = np.full(n, np.nan)
    s3_6h = np.full(n, np.nan)
    s4_6h = np.full(n, np.nan)
    
    for i in range(1, n):
        r1, r2, r3, r4, s1, s2, s3, s4 = camarilla_levels(high[i-1], low[i-1], close[i-1])
        r1_6h[i] = r1
        r2_6h[i] = r2
        r3_6h[i] = r3
        r4_6h[i] = r4
        s1_6h[i] = s1
        s2_6h[i] = s2
        s3_6h[i] = s3
        s4_6h[i] = s4
    
    # Calculate 6h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h Supertrend direction to 6h timeframe
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(direction_12h_aligned[i]) or 
            np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with 12h uptrend and volume
            if (close[i] > r3_6h[i] and 
                direction_12h_aligned[i] == 1 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with 12h downtrend and volume
            elif (close[i] < s3_6h[i] and 
                  direction_12h_aligned[i] == -1 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla S3
            if close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Camarilla R3
            if close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hSupertrend10_3_CamarillaR3S3_Breakout_Volume_Confirm"
timeframe = "6h"
leverage = 1.0