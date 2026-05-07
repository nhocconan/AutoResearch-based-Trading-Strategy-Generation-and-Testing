#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_TrendVolume_v1
Hypothesis: Weekly Donchian channel breakout on 1d timeframe with 1w trend filter and volume confirmation.
Captures long-term trends while avoiding whipsaws. Targets 10-20 trades/year per symbol to minimize fee drag.
Works in both bull and bear markets by using trend filter to align with higher timeframe direction.
"""

name = "1d_WeeklyDonchianBreakout_TrendVolume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channel (20-week period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period highest high and lowest low
    highest_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, highest_high)
    donchian_low = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # 1-week trend filter: EMA of weekly close
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND above weekly EMA with volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_1w_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low AND below weekly EMA with volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_1w_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below weekly Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above weekly Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals