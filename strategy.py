#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) from 6h for trend state, 1d EMA34 for higher timeframe trend alignment
# Volume confirmation (>1.5x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 6h.
# Williams Alligator catches emerging trends early, while 1d EMA34 filters counter-trend whipsaw in bear markets.
# Works in both bull and bear markets by only taking trades in direction of higher timeframe trend.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
    close_6h = df_6h['close'].values
    jaw = pd.Series(close_6h).ewm(span=13, adjust=False).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False).mean().values  # Smoothed
    teeth = pd.Series(close_6h).ewm(span=8, adjust=False).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False).mean().values  # Smoothed
    lips = pd.Series(close_6h).ewm(span=5, adjust=False).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False).mean().values  # Smoothed
    
    # Align Williams Alligator to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator lines aligned bullish (lips > teeth > jaw) + price above 1d EMA34 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator lines aligned bearish (lips < teeth < jaw) + price below 1d EMA34 + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines converge (teeth crosses below lips) OR price crosses below 1d EMA34
            if teeth_aligned[i] < lips_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines converge (teeth crosses above lips) OR price crosses above 1d EMA34
            if teeth_aligned[i] > lips_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals