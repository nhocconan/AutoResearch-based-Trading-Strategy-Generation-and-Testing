#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when green line > red line (bullish alignment) with 1d EMA50 uptrend and volume > 1.5x average
# Short when green line < red line (bearish alignment) with 1d EMA50 downtrend and volume > 1.5x average
# Exit when lines cross in opposite direction or volume drops below average
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend alignment, EMA for higher timeframe trend,
# and volume for conviction. Designed to capture sustained trends with low frequency.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5) on close prices
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips (5-period SMMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw.iloc[i]
        teeth_val = teeth.iloc[i]
        lips_val = lips.iloc[i]
        
        if position == 0:
            # Enter long: green line (lips) > red line (teeth) > blue line (jaw) with uptrend and volume
            if (lips_val > teeth_val > jaw_val and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: green line (lips) < red line (teeth) < blue line (jaw) with downtrend and volume
            elif (lips_val < teeth_val < jaw_val and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: lips crosses below teeth (trend weakening) or volume drops
            if (lips_val < teeth_val) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: lips crosses above teeth (trend weakening) or volume drops
            if (lips_val > teeth_val) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals