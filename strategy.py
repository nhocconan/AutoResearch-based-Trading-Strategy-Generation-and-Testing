#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume spike
# Williams Alligator uses three SMAs (Jaw: 13, Teeth: 8, Lips: 5) to identify trends.
# In uptrend: Lips > Teeth > Jaw (green alignment)
# In downtrend: Lips < Teeth < Jaw (red alignment)
# Trend filter: 1d EMA50 for higher timeframe bias
# Volume spike: volume > 1.5x 20-period average to filter weak moves
# Target: 12-37 trades/year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d Williams Alligator components
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # Jaw: 13-period SMMA (smoothed moving average)
    jaw_1d = close_1d.rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA
    teeth_1d = close_1d.rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA
    lips_1d = close_1d.rolling(window=5, min_periods=5).mean().values
    
    # Align Williams Alligator lines to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 1d EMA50 for trend filter
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Green alignment (Lips > Teeth > Jaw) + price above EMA50 + volume
        if (lips_1d_aligned[i] > teeth_1d_aligned[i] and 
            teeth_1d_aligned[i] > jaw_1d_aligned[i] and
            close[i] > ema50_1d_aligned[i] and
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Red alignment (Lips < Teeth < Jaw) + price below EMA50 + volume
        elif (lips_1d_aligned[i] < teeth_1d_aligned[i] and 
              teeth_1d_aligned[i] < jaw_1d_aligned[i] and
              close[i] < ema50_1d_aligned[i] and
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0