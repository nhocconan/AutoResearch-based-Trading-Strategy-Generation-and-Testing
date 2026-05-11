#!/usr/bin/env python3
"""
1d_WeeklyTrend_Breakout_Volume
Hypothesis: Daily breakouts above 20-day high or below 20-day low, filtered by weekly EMA50 trend direction and volume spikes. Designed to work in both bull and bear markets by following higher timeframe trend while using volatility-based breakouts for entry timing.
Target: 20-30 trades per year with clear trend-following signals.
"""

name = "1d_WeeklyTrend_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-day high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter (1w EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ratio[i])):
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
            # Long: price breaks above 20-day high + above weekly EMA50 + volume
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1d[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low + below weekly EMA50 + volume
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1d[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns below weekly EMA50 OR 20-day low
                if (close[i] < ema_50_1d[i]) or \
                   (close[i] < low_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns above weekly EMA50 OR 20-day high
                if (close[i] > ema_50_1d[i]) or \
                   (close[i] > high_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals