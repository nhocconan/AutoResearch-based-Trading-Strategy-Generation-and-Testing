#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_v1
# Hypothesis: Weekly Donchian channel breakout on daily timeframe with volume confirmation.
# Long when price breaks above weekly high (20-day weekly lookback) with above-average volume.
# Short when price breaks below weekly low with above-average volume.
# Weekly filter adapts to trends, works in bull/bear by capturing breakouts with momentum.
# Target: 15-25 trades/year (60-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels: 20-day high/low (approx 1 month)
    # Using 20-period rolling window on daily data
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 20-day average volume
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.2
        
        if position == 1:  # Long position
            # Exit: price closes below weekly low
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly high
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly high with volume confirmation
            if close[i] > high_20[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly low with volume confirmation
            elif close[i] < low_20[i] and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals