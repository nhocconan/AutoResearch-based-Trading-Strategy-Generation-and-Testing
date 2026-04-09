#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v2
# Hypothesis: 6h strategy using weekly pivot points for trend direction, Donchian(20) breakouts for entry timing, and volume confirmation.
# Weekly pivots provide institutional reference points; price tends to respect these levels.
# In bull markets: long when price > weekly pivot and breaks above Donchian high with volume.
# In bear markets: short when price < weekly pivot and breaks below Donchian low with volume.
# Volume confirmation filters false breakouts. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v2"
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
    
    # 1w HTF data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly resistance 1: R1 = 2*P - L
    weekly_r1 = 2 * weekly_pivot - low_1w
    # Weekly support 1: S1 = 2*P - H
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below weekly pivot or Donchian low
            if close[i] < weekly_pivot_aligned[i] or close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above weekly pivot or Donchian high
            if close[i] > weekly_pivot_aligned[i] or close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price above weekly pivot and breaks above Donchian high
                if close[i] > weekly_pivot_aligned[i] and close[i] > high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below weekly pivot and breaks below Donchian low
                elif close[i] < weekly_pivot_aligned[i] and close[i] < low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals