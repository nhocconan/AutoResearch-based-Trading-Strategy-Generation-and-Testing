#!/usr/bin/env python3
"""
6H_Donchian20_WeeklyTrend_VolumeConfirm
Hypothesis: 6h breakouts of 20-period Donchian channels with weekly trend filter (price > weekly SMA50) and volume confirmation. 
Weekly trend filter avoids counter-trend trades in strong trends, while volume confirmation ensures breakout validity.
Works in bull markets via breakout continuation and in bear markets via short breakdowns with trend alignment.
Targets 12-30 trades/year to stay within fee-efficient range for 6h timeframe.
"""
name = "6H_Donchian20_WeeklyTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate 6h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 6h volume > 1.8 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper with weekly uptrend and volume spike
            if (close[i] > high_max[i] and close[i-1] <= high_max[i-1] and 
                close[i] > sma_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with weekly downtrend and volume spike
            elif (close[i] < low_min[i] and close[i-1] >= low_min[i-1] and 
                  close[i] < sma_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion within channel)
            if position == 1 and close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals