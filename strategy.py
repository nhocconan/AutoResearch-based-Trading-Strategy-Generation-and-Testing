#!/usr/bin/env python3
# 6h_Donchian_20_WeeklyPivot_Direction_Volume
# Hypothesis: Uses 6h Donchian(20) breakout in the direction of the weekly pivot trend.
# Weekly pivot trend is determined by whether the weekly close is above/below the weekly pivot point (PP).
# Long when price breaks above Donchian upper band with weekly uptrend and volume confirmation.
# Short when price breaks below Donchian lower band with weekly downtrend and volume confirmation.
# Exit when price returns to the Donchian midpoint or trend reverses.
# Uses weekly pivot for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25.

name = "6h_Donchian_20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
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
    
    # Get weekly data for pivot and trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point (PP) and trend
    # PP = (H + L + C) / 3
    # Trend: close > PP = uptrend, close < PP = downtrend
    pp_w = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    pp_w_array = pp_w.values
    weekly_close = df_w['close'].values
    weekly_uptrend = weekly_close > pp_w_array  # True if weekly close above PP
    weekly_downtrend = weekly_close < pp_w_array  # True if weekly close below PP
    
    # Align weekly trend to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_w, weekly_downtrend)
    
    # Calculate Donchian(20) on 6h
    # Upper band = 20-period high
    # Lower band = 20-period low
    # Middle band = (upper + lower) / 2
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or \
           np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian upper band with weekly uptrend and volume confirmation
            if (close[i] > donchian_high[i] and 
                weekly_uptrend_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band with weekly downtrend and volume confirmation
            elif (close[i] < donchian_low[i] and 
                  weekly_downtrend_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midpoint or weekly trend turns down
            if (close[i] <= donchian_mid[i] or 
                not weekly_uptrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midpoint or weekly trend turns up
            if (close[i] >= donchian_mid[i] or 
                not weekly_downtrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals