#!/usr/bin/env python3
"""
4h Williams Alligator with 12h EMA50 Trend and Volume Confirmation
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence. 
When Alligator is 'sleeping' (lines intertwined) and then 'awakens' (lines diverge) 
with 12h uptrend/downtrend and volume spike, it signals trend start. 
Uses 4h timeframe with 12h HTF for trend. Targets 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 12h data for EMA50 trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator on 4h: SMAs of median price
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        alpha = 1.0 / period
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Align Alligator lines (no extra delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Calculate 20-period volume MA for 4h volume confirmation
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and volume MA
    start_idx = max(20, 13)  # 20 for volume MA, 13 for jaw
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        
        # Alligator sleeping: lines intertwined (max-min < 0.1% of price)
        alligator_range = max(jaw_val, teeth_val, lips_val) - min(jaw_val, teeth_val, lips_val)
        alligator_sleeping = alligator_range < (curr_close * 0.001)
        
        # Alligator awakening: lips outside jaw/teeth with separation
        lips_above = lips_val > max(jaw_val, teeth_val)
        lips_below = lips_val < min(jaw_val, teeth_val)
        lips_separation = abs(lips_val - (jaw_val + teeth_val) / 2) > (curr_close * 0.002)
        alligator_awakening = (lips_above or lips_below) and lips_separation
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator awakening AND lips above jaw/teeth AND price > EMA50 (uptrend) AND volume confirmation
            long_entry = (alligator_awakening and lips_above and 
                         curr_close > ema_trend and volume_confirm)
            # Short: Alligator awakening AND lips below jaw/teeth AND price < EMA50 (downtrend) AND volume confirmation
            short_entry = (alligator_awakening and lips_below and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator sleeping again OR lips cross below teeth OR price falls below EMA50
            if (alligator_sleeping or lips_val < teeth_val or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator sleeping again OR lips cross above teeth OR price rises above EMA50
            if (alligator_sleeping or lips_val > teeth_val or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0