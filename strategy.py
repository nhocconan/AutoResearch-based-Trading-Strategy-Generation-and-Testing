#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R4S4_Breakout_VolumeFilter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r4_1d = high_1d + 3 * (pivot_1d - low_1d)
    s4_1d = low_1d - 3 * (high_1d - pivot_1d)
    
    # Align all pivot levels
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1w: Calculate weekly pivot for trend direction ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Use weekly pivot as trend filter: price > weekly pivot = bullish, < = bearish
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # === 6h: ATR(20) for volatility and stop loss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # === 6h: Volume filter - current volume > 1.5x 20-period average ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        pivot_1w = pivot_1w_aligned[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        current_atr = atr[i]
        current_vol_ma = vol_ma[i]
        current_close = prices['close'].iloc[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(pivot_1w) or np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or
            np.isnan(r2) or np.isnan(s2) or np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or np.isnan(s4) or
            np.isnan(current_atr) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average volume
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        if position == 0:
            # Long conditions:
            # 1. Price above weekly pivot (bullish weekly bias)
            # 2. Price breaks above R4 with volume (strong breakout)
            if (current_close > pivot_1w and
                current_close > r4 and
                vol_condition):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below weekly pivot (bearish weekly bias)
            # 2. Price breaks below S4 with volume (strong breakdown)
            elif (current_close < pivot_1w and
                  current_close < s4 and
                  vol_condition):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below weekly pivot (trend change)
            # 2. Price hits S4 (strong support - take profit)
            # 3. ATR-based stop loss
            if (current_close < pivot_1w or
                current_close <= s4 or
                current_close < entry_price - 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above weekly pivot (trend change)
            # 2. Price hits R4 (strong resistance - take profit)
            # 3. ATR-based stop loss
            if (current_close > pivot_1w or
                current_close >= r4 or
                current_close > entry_price + 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals