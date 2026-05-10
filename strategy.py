#!/usr/bin/env python3
# 12h_Donchian_20_Breakout_1wTrend_Volume
# Hypothesis: Breakout of 20-period Donchian channel with weekly trend filter and volume confirmation.
# Long when: price breaks above 20-period high AND weekly uptrend AND volume > 1.5x average.
# Short when: price breaks below 20-period low AND weekly downtrend AND volume > 1.5x average.
# Uses weekly trend filter to avoid counter-trend trades, works in bull/bear by following higher timeframe trend.
# Target: 15-30 trades/year per symbol.

name = "12h_Donchian_20_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # 12h indicators
    # Donchian channel (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_rolling[i]) or np.isnan(low_rolling[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + Donchian breakout + volume confirmation
            if weekly_up and volume_confirm and close[i] > high_rolling[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + Donchian breakdown + volume confirmation
            elif weekly_down and volume_confirm and close[i] < low_rolling[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below midline or weekly trend changes
            midline = (high_rolling[i] + low_rolling[i]) / 2
            if close[i] < midline or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above midline or weekly trend changes
            midline = (high_rolling[i] + low_rolling[i]) / 2
            if close[i] > midline or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals