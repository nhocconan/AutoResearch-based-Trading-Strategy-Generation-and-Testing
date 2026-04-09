#!/usr/bin/env python3
# 6h_donchian_breakout_weekly_pivot_volume_v1
# Hypothesis: 6h strategy using Donchian(20) breakout filtered by weekly pivot direction and volume confirmation.
# In bull markets: long when price breaks above Donchian upper band AND weekly pivot bias is bullish (price > weekly pivot).
# In bear markets: short when price breaks below Donchian lower band AND weekly pivot bias is bearish (price < weekly pivot).
# Volume confirmation ensures breakouts have conviction. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years by requiring confluence of Donchian breakout + weekly pivot alignment + volume spike.
# Primary timeframe: 6h, HTF: 1d for Donchian and weekly pivot calculation (derived from 1d data).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Donchian channels and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate weekly pivot from prior week (using last 5 trading days approximation)
    if len(df_1d) >= 5:
        # Use last 5 days for weekly high/low/close
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot point: (High + Low + Close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves back below Donchian low or volume dries up
            if close[i] < donchian_low_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves back above Donchian high or volume dries up
            if close[i] > donchian_high_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above Donchian high AND price above weekly pivot (bullish bias)
                if close[i] > donchian_high_aligned[i] and close[i] > weekly_pivot_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND price below weekly pivot (bearish bias)
                elif close[i] < donchian_low_aligned[i] and close[i] < weekly_pivot_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals