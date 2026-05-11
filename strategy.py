#!/usr/bin/env python3
# 1D_Weekly_RangeBreakout_v1
# Hypothesis: Use weekly price range (high-low) to detect consolidation periods.
# When price breaks above the weekly high with volume confirmation, go long.
# When price breaks below the weekly low with volume confirmation, go short.
# Weekly timeframe provides structure, daily provides entry timing.
# Designed for low trade frequency by requiring both range breakout and volume spike.
# Works in both bull and bear markets by following the intermediate-term trend from weekly close.

name = "1D_Weekly_RangeBreakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for range
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly High and Low
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Align weekly levels to daily
    high_weekly_aligned = align_htf_to_ltf(prices, df_weekly, high_weekly)
    low_weekly_aligned = align_htf_to_ltf(prices, df_weekly, low_weekly)
    
    # Volume Spike Detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_weekly_aligned[i]) or np.isnan(low_weekly_aligned[i]) or
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above weekly high with volume
            if (close[i] > high_weekly_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low with volume
            elif (close[i] < low_weekly_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout
            if position == 1:
                # Exit long: price breaks below weekly low
                if close[i] < low_weekly_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above weekly high
                if close[i] > high_weekly_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals