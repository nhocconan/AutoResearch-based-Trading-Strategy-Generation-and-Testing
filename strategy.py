#!/usr/bin/env python3
"""
12h_WilliamsAlligator_ElderRay_Trend_VolumeConfirm
Hypothesis: Use Williams Alligator (JAW/TEETH/LIPS) for trend direction on 12h timeframe,
combined with Elder Ray (Bull/Bear Power) from 1d for momentum confirmation.
Add volume spike (>2x 20-period average) as entry filter.
Go long when price > Alligator teeth (TEETH) AND Bull Power > 0 AND volume confirmation.
Go short when price < Alligator teeth AND Bear Power < 0 AND volume confirmation.
Exit on opposite Alligator TEETH crossover.
Designed for low-frequency trading (target 12-30 trades/year) with strong trend/momentum alignment
to work in both bull and bear markets by avoiding choppy conditions.
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
    
    # Williams Alligator from 12h timeframe (SMMA = smoothed moving average)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # SMMA calculation (similar to Wilder's smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_VALUE) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_12h = df_12h['close'].values
    jaw = smma(close_12h, 13)  # BLUE line
    teeth = smma(close_12h, 8)  # RED line
    lips = smma(close_12h, 5)   # GREEN line
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Elder Ray from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # EMA13 for Elder Ray calculation
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = df_1d['low'].values - ema13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(teeth_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        teeth_val = teeth_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price above TEETH (Alligator uptrend) AND Bull Power positive AND volume confirmation
            if close[i] > teeth_val and bull_val > 0 and vol_conf:
                signals[i] = size
                position = 1
            # Short: price below TEETH (Alligator downtrend) AND Bear Power negative AND volume confirmation
            elif close[i] < teeth_val and bear_val < 0 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below TEETH (trend change)
            if close[i] < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above TEETH (trend change)
            if close[i] > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0