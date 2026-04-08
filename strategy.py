#!/usr/bin/env python3
# 6h_alligator_elder_ray_v1
# Hypothesis: Combines Elder Ray (Bull/Bear Power) with Williams Alligator on 6h timeframe to identify trend strength and exhaustion points. 
# Uses 1d timeframe for trend filter to avoid counter-trend trades. 
# Long when: Bull Power > 0, Bear Power < 0, price > Alligator's Jaw (13-period SMMA), and price > 1d EMA50
# Short when: Bear Power < 0, Bull Power < 0, price < Alligator's Jaw, and price < 1d EMA50
# Exit when Elder Ray signals weaken or price crosses Alligator's Teeth (8-period SMMA)
# Designed to work in both bull and bear markets by requiring alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def smma(series, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components (all SMMA)
    jaw = smma(close, 13)   # Blue line - 13-period SMMA
    teeth = smma(close, 8)  # Red line - 8-period SMMA
    lips = smma(close, 5)   # Green line - 5-period SMMA
    
    # Elder Ray components
    # Bull Power = High - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period for all indicators
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 OR price crosses below Teeth (8-period SMMA)
            if (bull_power[i] <= 0) or (close[i] < teeth[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 OR price crosses above Teeth (8-period SMMA)
            if (bear_power[i] >= 0) or (close[i] > teeth[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Bull Power > 0, Bear Power < 0, price > Jaw, price > 1d EMA50
            if (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > jaw[i]) and (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0, Bull Power < 0, price < Jaw, price < 1d EMA50
            elif (bear_power[i] < 0) and (bull_power[i] < 0) and (close[i] < jaw[i]) and (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals