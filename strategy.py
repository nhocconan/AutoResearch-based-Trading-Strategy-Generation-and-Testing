#!/usr/bin/env python3
# 6H_Range_Breakout_Volume
# Hypothesis: In 6h timeframe, trade breakouts from daily-defined ranges with volume confirmation.
# Uses 1-day high-low range to define support/resistance. Breakouts occur when price closes outside
# the prior day's range with volume > 1.5x average. Uses volume to filter false breakouts.
# Works in both bull and break as it captures momentum moves regardless of direction.
# Target: 15-25 trades/year per symbol.

name = "6H_Range_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # 6h indicators
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily range calculation (prior day's high-low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Prior day's range (shifted by 1 to avoid look-ahead)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    # First day has no prior, set to same day
    prior_high[0] = high_1d[0]
    prior_low[0] = low_1d[0]
    
    # Align prior day's range to 6h
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(prior_high_aligned[i]) or np.isnan(prior_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: break above prior day's high with volume
            if close[i] > prior_high_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: break below prior day's low with volume
            elif close[i] < prior_low_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to prior day's range or volume weakens
            if close[i] < prior_high_aligned[i] or volume_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to prior day's range or volume weakens
            if close[i] > prior_low_aligned[i] or volume_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals