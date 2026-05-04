#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (jaw/teeth/lips) identifies trending vs ranging markets.
# In trends (lips > teeth > jaw for uptrend, inverse for downtrend), we enter breakouts.
# 1d EMA50 ensures alignment with daily trend. Volume spike (>2x 20 EMA) confirms participation.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 50-150 trades over 4 years.

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams Alligator on 6h timeframe
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    close_series = pd.Series(close)
    jaw = close_series.ewm(alpha=1/13, adjust=False).mean().shift(8)
    teeth = close_series.ewm(alpha=1/8, adjust=False).mean().shift(5)
    lips = close_series.ewm(alpha=1/5, adjust=False).mean().shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(jaw_values[i]) or 
            np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        # Alligator conditions
        # Uptrend: lips > teeth > jaw
        # Downtrend: lips < teeth < jaw
        lips_gt_teeth = lips_values[i] > teeth_values[i]
        teeth_gt_jaw = teeth_values[i] > jaw_values[i]
        uptrend = lips_gt_teeth and teeth_gt_jaw
        
        lips_lt_teeth = lips_values[i] < teeth_values[i]
        teeth_lt_jaw = teeth_values[i] < jaw_values[i]
        downtrend = lips_lt_teeth and teeth_lt_jaw
        
        if position == 0:
            # Long conditions: uptrend + price above lips + volume spike
            if uptrend and close[i] > lips_values[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: downtrend + price below lips + volume spike
            elif downtrend and close[i] < lips_values[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend changes or price crosses below teeth
            if not uptrend or close[i] < teeth_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend changes or price crosses above teeth
            if not downtrend or close[i] > teeth_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals