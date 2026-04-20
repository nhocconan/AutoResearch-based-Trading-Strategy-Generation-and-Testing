#!/usr/bin/env python3
# 1d_1w_Donchian_Breakout_Trend
# Hypothesis: Trade breakouts from weekly Donchian channels on daily timeframe with trend filter.
# Uses weekly Donchian(20) high/low as breakout levels and daily EMA(50) for trend filter.
# Volume confirmation ensures breakout validity. Designed for 7-25 trades per year by requiring
# multiple confirmations and using higher timeframe structure. Works in both bull and bear markets
# by only taking breakouts in the direction of the weekly trend.

name = "1d_1w_Donchian_Breakout_Trend"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian high and low
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA for trend filter (50-period)
    ema_50 = pd.Series(close_1w := df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily volume average for spike confirmation (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume surge and above weekly EMA
            if (close[i] > donchian_high_aligned[i] * 1.001 and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume surge and below weekly EMA
            elif (close[i] < donchian_low_aligned[i] * 0.999 and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals