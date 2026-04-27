#!/usr/bin/env python3
"""
4h_Williams_Alligator_Trend_With_Volume_Spike
Hypothesis: Use Williams Alligator (13/8/5 SMAs) for trend direction combined with volume spike (>3x 20-bar average) to filter entries. 
Long when price > Alligator Jaw and lips above teeth; short when price < Jaw and lips below teeth. 
Exit when price crosses back through Jaw or trend weakens. Volume confirmation prevents whipsaws in ranging markets.
Target 20-30 trades/year to avoid fee drag while capturing strong trends.
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
    
    # Get daily data for Williams Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-bar SMMA), Teeth (8-bar SMMA), Lips (5-bar SMMA)
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(df_1d['close'].values, 13)
    teeth = smma(df_1d['close'].values, 8)
    lips = smma(df_1d['close'].values, 5)
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 3.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 3.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Alligator (13 periods) and volume MA (20)
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price above Jaw AND lips above teeth (bullish alignment) + volume spike
            if close[i] > jaw_val and lips_val > teeth_val and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price below Jaw AND lips below teeth (bearish alignment) + volume spike
            elif close[i] < jaw_val and lips_val < teeth_val and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Jaw OR lips cross below teeth (trend weakening)
            if close[i] < jaw_val or lips_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Jaw OR lips cross above teeth (trend weakening)
            if close[i] > jaw_val or lips_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Williams_Alligator_Trend_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0