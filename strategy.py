#!/usr/bin/env python3
"""
4h_CamarillaPivot_R1S1_Breakout_Volume_Conservative_v1
Concept: 4h price breakout at Camarilla R1/S1 levels with daily volume confirmation and 12h trend filter.
- Long: Price > 4h Camarilla R1 AND daily volume > 1.5x 20-period avg AND 12h close > 12h open (bullish candle)
- Short: Price < 4h Camarilla S1 AND daily volume > 1.5x 20-period avg AND 12h close < 12h open (bearish candle)
- Exit: Price crosses back through 4h Camarilla Pivot point
- Position sizing: 0.25
- Target: 75-200 total trades over 4 years
- Uses Camarilla levels from prior 1d for structure, volume for conviction, 12h for trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_CamarillaPivot_R1S1_Breakout_Volume_Conservative_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla levels from prior day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = camarilla_pivot + (range_1d * 1.0 / 12.0)
    camarilla_s1 = camarilla_pivot - (range_1d * 1.0 / 12.0)
    
    # Align Camarilla levels to 4h (use prior day's levels)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # === 12h: Trend Filter (bullish/bearish candle) ===
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    bullish_12h = close_12h > open_12h  # True if bullish candle
    bearish_12h = close_12h < open_12h   # True if bearish candle
    bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_12h.astype(float))
    bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_12h.astype(float))
    
    # === 4h: Price data ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        cp = camarilla_pivot_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        curr_vol = volume_1d_aligned[i]
        bullish = bullish_12h_aligned[i]
        bearish = bearish_12h_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(cp) or np.isnan(r1) or np.isnan(s1) or 
            np.isnan(vol_ma) or np.isnan(curr_vol) or
            np.isnan(bullish) or np.isnan(bearish)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-period average
        vol_condition = curr_vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above R1 with bullish 12h and volume spike
            if close[i] > r1 and bullish > 0.5 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with bearish 12h and volume spike
            elif close[i] < s1 and bearish > 0.5 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Camarilla pivot
            if close[i] < cp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Camarilla pivot
            if close[i] > cp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals