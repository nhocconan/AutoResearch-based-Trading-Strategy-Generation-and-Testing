#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Long when green line > red line (bullish alignment) with EMA50 uptrend and volume > 1.5x average
# Short when green line < red line (bearish alignment) with EMA50 downtrend and volume > 1.5x average
# Exit when lines cross in opposite direction or volume drops below average
# Williams Alligator uses smoothed medians (Jaw=13, Teeth=8, Lips=5) to identify trends
# Designed to capture sustained trends with low frequency, suitable for 12h timeframe
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "12h_Williams_Alligator_1dEMA50_Volume"
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
    
    # Calculate Williams Alligator lines (using median prices)
    median_price = (high + low) / 2
    
    # Jaw (blue line) - 13-period SMMA of median, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth (red line) - 8-period SMMA of median, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips (green line) - 5-period SMMA of median, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), teeth.values)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': median_price}), lips.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: lips > teeth > jaw (bullish alignment), EMA50 up, volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: lips < teeth < jaw (bearish alignment), EMA50 down, volume confirmation
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: alignment breaks (lips < teeth) or volume drops
            if (lips_aligned[i] < teeth_aligned[i]) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment breaks (lips > teeth) or volume drops
            if (lips_aligned[i] > teeth_aligned[i]) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals