#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_4HTrend_Filter_v1
Concept: Daily pivot breakout on 12h with 4H EMA trend filter to improve win rate in both bull and bear markets.
- Long when price breaks above R1 and 4H EMA20 is rising (trend filter)
- Short when price breaks below S1 and 4H EMA20 is falling
- Exit when price crosses back to previous day's close
- Uses volume confirmation (vol > 1.5x average) to filter false breakouts
- Conservative sizing (0.25) to manage drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_4HTrend_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4H data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # === Daily Pivots ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    
    # Align daily pivots to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # === 4H EMA20 trend filter ===
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === 12h Volume ratio ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for EMA20
    
    for i in range(start_idx, n):
        # Get values
        ema20_val = ema20_4h_aligned[i]
        close_val = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        prev_close_val = prev_close_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema20_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(prev_close_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trend direction from 4H EMA slope
            ema20_prev = ema20_4h_aligned[i-1]
            ema_rising = ema20_val > ema20_prev
            ema_falling = ema20_val < ema20_prev
            
            # Long: Break above R1 with volume and rising 4H EMA
            if close_val > r1_val and vol_ratio_val > 1.5 and ema_rising:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume and falling 4H EMA
            elif close_val < s1_val and vol_ratio_val > 1.5 and ema_falling:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below previous day's close
            if close_val <= prev_close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above previous day's close
            if close_val >= prev_close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals