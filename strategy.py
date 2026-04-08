#!/usr/bin/env python3
"""
4h_donchian_breakout_volume_v2
Hypothesis: Donchian breakout with volume confirmation and weekly trend filter.
- Only trade breakouts in direction of weekly trend (above/below weekly 20-period SMA)
- Long: Price breaks above 20-bar Donchian high + volume > 1.5x average + weekly uptrend
- Short: Price breaks below 20-bar Donchian low + volume > 1.5x average + weekly downtrend
- Exit when price crosses 10-bar SMA in opposite direction
- Target: 20-40 trades/year to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period SMA for exit
    sma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Average volume (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter (20-period SMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    weekly_uptrend = close > sma_20_1w_aligned
    weekly_downtrend = close < sma_20_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(sma_10[i]) or np.isnan(avg_volume[i]) or
            np.isnan(weekly_uptrend[i]) or np.isnan(weekly_downtrend[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price crosses below 10 SMA or weekly trend turns down
            if close[i] < sma_10[i] or weekly_downtrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 10 SMA or weekly trend turns up
            if close[i] > sma_10[i] or weekly_uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: breakout above Donchian high + volume spike + weekly uptrend
            if (high[i] > donchian_high[i] and 
                volume[i] > 1.5 * avg_volume[i] and
                weekly_uptrend[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low + volume spike + weekly downtrend
            elif (low[i] < donchian_low[i] and 
                  volume[i] > 1.5 * avg_volume[i] and
                  weekly_downtrend[i]):
                position = -1
                signals[i] = -0.25
    
    return signals