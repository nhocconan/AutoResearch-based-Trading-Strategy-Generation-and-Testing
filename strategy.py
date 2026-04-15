#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price reversal at 12h Bollinger Bands with volume confirmation
# Uses mean reversion at Bollinger Band extremes (20, 2) on 12h timeframe,
# confirmed by volume spike on 4h. Works in ranging markets (both bull and bear)
# by fading extreme moves. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for volume and price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 12h data for Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Bollinger Bands (20, 2) on 12h
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i])):
            continue
        
        # Long entry: price touches lower Bollinger Band + volume spike
        if (low[i] <= lower_bb_aligned[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches upper Bollinger Band + volume spike
        elif (high[i] >= upper_bb_aligned[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to middle (SMA) or opposite band touch
        elif position == 1 and (high[i] >= sma_20[i] if not np.isnan(sma_20[i]) else False):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= sma_20[i] if not np.isnan(sma_20[i]) else False):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_BollingerReversal_Volume_12h"
timeframe = "4h"
leverage = 1.0