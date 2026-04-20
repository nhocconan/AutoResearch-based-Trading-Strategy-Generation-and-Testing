#!/usr/bin/env python3
# 12h_Donchian20_Trend_Strategy
# Hypothesis: 12h Donchian(20) breakout captures multi-day momentum.
# Filtered by 1d EMA200 to align with daily trend and avoid counter-trend trades.
# Uses 1d volume spike (volume > 1.5x 20-period average) for institutional confirmation.
# Exits when price closes below/above Donchian(10) or trend reverses.
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag.
# Works in bull via breakouts, in bear via trend-following on shorts.

name = "12h_Donchian20_Trend_Strategy"
timeframe = "12h"
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
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d volume spike confirmation
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma20_1d * 1.5)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian(10) for exit
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1d EMA200
            uptrend = close[i] > ema200_1d_aligned[i]
            downtrend = close[i] < ema200_1d_aligned[i]
            
            # Long: uptrend + price breaks above Donchian(20) high + volume spike
            if uptrend and high[i] > donchian_high_20[i] and vol_spike_1d_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price breaks below Donchian(20) low + volume spike
            elif downtrend and low[i] < donchian_low_20[i] and vol_spike_1d_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price closes below Donchian(10) low or trend reverses
            if close[i] < donchian_low_10[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price closes above Donchian(10) high or trend reverses
            if close[i] > donchian_high_10[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals