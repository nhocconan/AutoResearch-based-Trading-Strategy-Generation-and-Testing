#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d daily close above/below weekly SMA50 with volume confirmation.
# Long when close > weekly SMA50 and volume > 1.5x 20-day average.
# Short when close < weekly SMA50 and volume > 1.5x 20-day average.
# Exit when close crosses back below/above weekly SMA50.
# Weekly SMA50 provides strong trend filter for daily entries.
# Volume confirms institutional participation.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_WeeklySMA50_Volume_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Weekly data for SMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # SMA50 on weekly close
    sma_50 = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly SMA50 to daily timeframe
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for weekly SMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(sma_50_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: close above weekly SMA50 and volume filter
            long_cond = (close[i] > sma_50_aligned[i]) and volume_filter[i]
            # Short conditions: close below weekly SMA50 and volume filter
            short_cond = (close[i] < sma_50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close crosses below weekly SMA50
            if close[i] < sma_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close crosses above weekly SMA50
            if close[i] > sma_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals