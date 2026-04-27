#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In trending markets: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: three SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # Blue line (13-period)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # Red line (8-period)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # Green line (5-period)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
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
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trending market logic with Williams Alligator and volume filter
        if close[i] > ema50_1d_aligned[i] and volume_filter[i]:  # Uptrend filter
            # Alligator aligned for uptrend: Lips > Teeth > Jaw
            if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                signals[i] = 0.25
                position = 1
            # Exit long when Alligator loses alignment
            elif position == 1 and not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
        elif close[i] < ema50_1d_aligned[i] and volume_filter[i]:  # Downtrend filter
            # Alligator aligned for downtrend: Lips < Teeth < Jaw
            if lips[i] < teeth[i] and teeth[i] < jaw[i]:
                signals[i] = -0.25
                position = -1
            # Exit short when Alligator loses alignment
            elif position == -1 and not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
        else:
            # Hold current position or stay flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0