#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_Trend_Strategy
Hypothesis: Combines ADX trend strength with Williams Alligator (smoothed medians) to capture strong 6h trends while avoiding chop. Uses 1d ADX > 25 for trend regime, and Alligator (Jaw/Teeth/Lips) alignment for entry/exit. Works in bull/bear by following trend direction. Discrete sizing (0.25) minimizes fee drag. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 6h data (smoothed medians)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    median_price = (high + low) / 2
    
    def smma(data, period):
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN for validity check
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need ADX (14+14=28) and Alligator (max shift 8)
    start_idx = max(28, 8)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        close_val = close[i]
        
        if position == 0:
            # Look for entry: ADX > 25 (strong trend) + Alligator alignment
            # Long: ADX > 25 AND Lips > Teeth > Jaw (bullish alignment)
            # Short: ADX > 25 AND Lips < Teeth < Jaw (bearish alignment)
            if adx_val > 25:
                if lips_val > teeth_val > jaw_val:
                    signals[i] = size
                    position = 1
                elif lips_val < teeth_val < jaw_val:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit: Alligator loses bullish alignment OR ADX weakens (< 20)
            if not (lips_val > teeth_val > jaw_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: Alligator loses bearish alignment OR ADX weakens (< 20)
            if not (lips_val < teeth_val < jaw_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_WilliamsAlligator_Trend_Strategy"
timeframe = "6h"
leverage = 1.0