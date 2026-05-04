#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) from completed 12h for structure, 1d EMA34 for trend filter
# Volume confirmation (>1.8x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h.
# Williams Alligator provides clear trend definition with built-in smoothing, reducing whipsaw in both bull and bear markets.
# 1d EMA34 provides stronger trend filter than 12h EMA34, reducing false signals during sideways markets.

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Calculate Williams Alligator lines (Smoothed Medians)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_12h = pd.Series(median_12h).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    jaw_12h_shifted = np.roll(jaw_12h, 8)
    jaw_12h_shifted[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_12h = pd.Series(median_12h).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    teeth_12h_shifted = np.roll(teeth_12h, 5)
    teeth_12h_shifted[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips_12h = pd.Series(median_12h).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    lips_12h_shifted = np.roll(lips_12h, 3)
    lips_12h_shifted[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) + price above EMA34 + volume spike
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) + price below EMA34 + volume spike
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines cross (Lips < Teeth) OR price crosses below EMA34
            if lips_aligned[i] < teeth_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross (Lips > Teeth) OR price crosses above EMA34
            if lips_aligned[i] > teeth_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals