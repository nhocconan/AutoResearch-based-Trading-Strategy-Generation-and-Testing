#!/usr/bin/env python3
"""
4h_Vortex_Trend_Filter_V1
Hypothesis: Uses Vortex indicator (VI+ and VI-) to determine trend direction on 1-day timeframe, 
combined with 4-hour price action and volume confirmation. Enters long when VI+ > VI- (uptrend) 
and price breaks above 4-hour high of prior period with volume spike; shorts when VI- > VI+ 
(downtrend) and price breaks below 4-hour low with volume spike. Uses ATR-based stop via 
signal=0 when trend reverses. Designed for low frequency (<40 trades/year) with strong 
performance in trending markets while avoiding whipsaws in ranging conditions.
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
    
    # Get 1d data for Vortex trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Vortex indicator (VI+ and VI-) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(abs(high_1d[1:] - close_1d[:-1]), 
                               abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Vortex movements
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])  # |current high - prior low|
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])  # |current low - prior high|
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Smooth over 14 periods
    def ema_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(arr[1:period])  # Skip index 0 (nan)
        alpha = 2 / (period + 1)
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]):
                result[i] = arr[i] * alpha + result[i-1] * (1 - alpha)
        return result
    
    vi_plus = ema_smooth(vm_plus, 14)
    vi_minus = ema_smooth(vm_minus, 14)
    tr14 = ema_smooth(tr, 14)
    
    # Normalize to get VI+ and VI-
    vi_plus_norm = vi_plus / tr14
    vi_minus_norm = vi_minus / tr14
    
    # Get 4h data for price action and volume
    df_4h = get_htf_data(prices, '4h')  # For reference only, we use prices directly
    
    # Calculate 4-period high/low for breakout detection
    high_4 = np.full(n, np.nan)
    low_4 = np.full(n, np.nan)
    for i in range(4, n):
        high_4[i] = np.max(high[i-4:i])
        low_4[i] = np.min(low[i-4:i])
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 1d Vortex indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus_norm)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus_norm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(high_4[i]) or np.isnan(low_4[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (uptrend) + break above 4-period high + volume spike
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                close[i] > high_4[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (downtrend) + break below 4-period low + volume spike
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  close[i] < low_4[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend turns down (VI- > VI+)
            if vi_minus_aligned[i] > vi_plus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend turns up (VI+ > VI-)
            if vi_plus_aligned[i] > vi_minus_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Vortex_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0