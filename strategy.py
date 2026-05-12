#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: Use Donchian Channel breakout on 12h with 1d EMA trend filter and volume confirmation.
# Enter long when price breaks above upper Donchian band (20-period high) with volume, short when breaks below lower band (20-period low) with volume, only in direction of 1d trend.
# Exit when price returns to midpoint of Donchian channel or trend reverses.
# Designed for low frequency (12-37 trades/year) by using 12h for signal and 1d for trend filter.

name = "12h_Donchian20_Breakout_1dTrend_Volume"
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
    
    # === 12h data for Donchian Channel ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian Channel (20-period high/low)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian components to 12h (wait for 12h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) on 1d for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian with volume, in uptrend
            if close[i] > donchian_high_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian with volume, in downtrend
            elif close[i] < donchian_low_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to midpoint or trend reverses
            if close[i] <= donchian_mid_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to midpoint or trend reverses
            if close[i] >= donchian_mid_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals