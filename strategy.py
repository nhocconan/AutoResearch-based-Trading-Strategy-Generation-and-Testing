#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Trend_WeeklyFilter_VolumeConfirm
Hypothesis: Uses Williams Alligator (Jaw/Teeth/Lips) on 1d for trend direction, 
filtered by 1w EMA50 for higher timeframe alignment. Enters on Alligator alignment 
with volume confirmation (>1.5x 20-period avg). Exits when Alligator reverses or 
price crosses Jaw. Designed for low trade frequency (<25/year) to minimize fee drag 
and work in both bull/bear markets via trend filter and weekly confirmation.
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
    
    # Get 1d data for Alligator and weekly EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator on 1d: Jaw (13,8), Teeth (8,5), Lips (5,3) - SMMA
    def smma(source, length):
        # Smoothed Moving Average: first value is SMA, then recursive
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < length:
            return result
        # Initial SMA
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA*(length-1) + PRICE) / length
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)  # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    # Align Alligator lines to 1d timeframe (no additional delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Weekly EMA50 for trend filter (needs completed weekly bar)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Alligator (max 13), weekly EMA50 (50), volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_wk_val = ema_50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Alligator aligned (Lips > Teeth > Jaw for long, reverse for short)
            # Plus weekly trend filter and volume confirmation
            bullish_alligator = (lips_val > teeth_val) and (teeth_val > jaw_val)
            bearish_alligator = (lips_val < teeth_val) and (teeth_val < jaw_val)
            
            long_condition = bullish_alligator and (close_val > ema_wk_val) and vol_conf
            short_condition = bearish_alligator and (close_val < ema_wk_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when Alligator reverses (Lips < Teeth) OR price crosses below Jaw
            exit_condition = (lips_val < teeth_val) or (close_val < jaw_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when Alligator reverses (Lips > Teeth) OR price crosses above Jaw
            exit_condition = (lips_val > teeth_val) or (close_val > jaw_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsAlligator_Trend_WeeklyFilter_VolumeConfirm"
timeframe = "1d"
leverage = 1.0