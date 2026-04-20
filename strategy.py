#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_With_1D_Trend_Filter
Hypothesis: Use TRIX momentum oscillator with volume spikes and 1-day trend filter.
Long when TRIX crosses above zero with volume spike and 1-day uptrend; short when TRIX crosses below zero with volume spike and 1-day downtrend.
TRIX (TRIple Exponential Average) filters out market noise and identifies significant momentum shifts.
Volume spike (2x 20-period average) confirms breakout strength. 1-day EMA50 trend filter avoids counter-trend trades.
Designed for 4h timeframe to target 50-120 trades over 4 years (12-30/year) with position size 0.25.
Works in bull/bear markets: trend filter prevents counter-trend trades, momentum captures sustained moves.
"""

name = "4h_TRIX_Volume_Spike_With_1D_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average"""
    result = np.full_like(values, np.nan)
    if len(values) >= period:
        multiplier = 2.0 / (period + 1)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
    return result

def trix(values, period):
    """Calculate TRIX (TRIple Exponential Average) momentum oscillator"""
    if len(values) < period * 3:
        return np.full_like(values, np.nan)
    # Triple EMA
    ema1 = ema(values, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    # Calculate TRIX as percentage rate of change of triple EMA
    trix_val = np.full_like(values, np.nan)
    for i in range(1, len(ema3)):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix_val[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    return trix_val

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate TRIX on price (period=12)
    trix_val = trix(close, 12)
    
    # Calculate volume spike (volume > 2x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_val[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike AND 1-day uptrend
            if trix_val[i] > 0 and trix_val[i-1] <= 0 and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike AND 1-day downtrend
            elif trix_val[i] < 0 and trix_val[i-1] >= 0 and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero OR 1-day trend turns down
            if trix_val[i] < 0 and trix_val[i-1] >= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero OR 1-day trend turns up
            if trix_val[i] > 0 and trix_val[i-1] <= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals