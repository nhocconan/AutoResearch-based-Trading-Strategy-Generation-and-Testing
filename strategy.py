#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian breakout with 1-day volume filter and trend filter.
Long when price breaks above 20-period high, 1-day volume > 20-period average, and 1-day EMA50 rising.
Short when price breaks below 20-period low, 1-day volume > 20-period average, and 1-day EMA50 falling.
Exit when price reverses to touch the opposite Donchian band (mean reversion) or trend fails.
Donchian provides clear breakout levels; volume filter ensures participation; 1-day EMA50 filters trend.
Designed for low trade frequency by requiring three confirmations. Works in bull/bear by following daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 and volume filters - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1-day volume average (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 6-hour Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 20-period high, volume above average, EMA50 rising
            if (high[i] > high_20[i-1] and  # Breakout condition
                volume[i] > vol_avg_1d_aligned[i] and  # Volume confirmation
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):  # Uptrend
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-period low, volume above average, EMA50 falling
            elif (low[i] < low_20[i-1] and  # Breakdown condition
                  volume[i] > vol_avg_1d_aligned[i] and  # Volume confirmation
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):  # Downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price touches or goes below lower Donchian band OR trend fails
                if (low[i] <= low_20[i] or 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price touches or goes above upper Donchian band OR trend fails
                if (high[i] >= high_20[i] or 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_Breakout_1dVol_EMA50_Trend"
timeframe = "6h"
leverage = 1.0