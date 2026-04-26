#!/usr/bin/env python3
"""
6h_WeeklyPivot_Camarilla_Breakout_v1
Hypothesis: Trade 6h breakouts at Camarilla R4/S4 levels (strong breakout) with weekly pivot direction filter for trend alignment. Uses volume confirmation (1.5x average) to filter false breakouts. Designed for low trade frequency (~10-25/year) by requiring confluence: Camarilla extreme breakout + weekly trend + volume spike. Weekly pivot provides robust trend filter that works in both bull (buy above weekly pivot) and bear (sell below weekly pivot) markets. Target symbols: BTC/ETH.
"""

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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical Camarilla: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla multipliers
    # R4 = close + 1.5 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_s4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot points (standard calculation)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # We'll use previous completed weekly bar for pivot
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    weekly_close = df_1w['close'].shift(1).values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align HTF indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: 1.5x average volume (moderate filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*6h = 6 days
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d lookback (1), 1w lookback (1), volume MA (24)
    start_idx = max(1, 1, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        camarilla_r4_val = camarilla_r4_aligned[i]
        camarilla_s4_val = camarilla_s4_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: break above Camarilla R4, above weekly pivot, volume spike
            long_signal = (high_val > camarilla_r4_val) and (close_val > weekly_pivot_val) and (volume_val > 1.5 * vol_ma_val)
            # Short: break below Camarilla S4, below weekly pivot, volume spike
            short_signal = (low_val < camarilla_s4_val) and (close_val < weekly_pivot_val) and (volume_val > 1.5 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below weekly pivot (trend change)
            if close_val < weekly_pivot_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above weekly pivot (trend change)
            if close_val > weekly_pivot_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Camarilla_Breakout_v1"
timeframe = "6h"
leverage = 1.0