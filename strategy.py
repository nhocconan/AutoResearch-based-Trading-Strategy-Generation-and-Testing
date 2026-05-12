#!/usr/bin/env python3
# 12h_Donchian_20_Breakout_1wTrend_Volume
# Hypothesis: Donchian channel breakout on 12h with 1w trend filter and volume confirmation.
# The 1w Donchian trend ensures we trade in the direction of the weekly trend.
# Volume confirmation adds conviction to breakouts. Designed for low trade frequency
# to minimize fee drag while capturing major moves in both bull and bear markets.

name = "12h_Donchian_20_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # === 12h Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1w Trend Filter: Donchian (20) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w Donchian for trend direction
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    trend_up = donchian_high_1w > donchian_high_1w[-1]  # New high
    trend_down = donchian_low_1w < donchian_low_1w[-1]  # New low
    
    # Align 1w trend to 12h
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]
        breakout_down = close[i] < donchian_low[i-1]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: upward breakout, uptrend, volume
            if breakout_up and trend_up_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: downward breakout, downtrend, volume
            elif breakout_down and trend_down_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: downward breakout or trend reversal
            if breakout_down or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: upward breakout or trend reversal
            if breakout_up or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals