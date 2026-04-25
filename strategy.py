#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot (price > weekly pivot = long bias, < weekly pivot = short bias) with volume confirmation (>1.5x 20-bar average). Weekly pivot provides structural bias from higher timeframe, Donchian captures breakouts, volume filters false breakouts. Designed for ~12-37 trades/year by requiring strong breakouts, HTF bias alignment, and volume confirmation. Works in bull/bear markets via weekly pivot filter; avoids whipsaws via volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation (previous completed week)
    weekly_high = align_htf_to_ltf(prices, df_1w, df_1w['high'].values, additional_delay_bars=1)
    weekly_low = align_htf_to_ltf(prices, df_1w, df_1w['low'].values, additional_delay_bars=1)
    weekly_close = align_htf_to_ltf(prices, df_1w, df_1w['close'].values, additional_delay_bars=1)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: >1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, lookback)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Determine bias from weekly pivot
            bullish_bias = close[i] > weekly_pivot[i]
            bearish_bias = close[i] < weekly_pivot[i]
            
            # Long: Donchian breakout above resistance with bullish bias and volume
            long_signal = (close[i] > highest_high[i]) and bullish_bias and vol_confirm[i]
            # Short: Donchian breakdown below support with bearish bias and volume
            short_signal = (close[i] < lowest_low[i]) and bearish_bias and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel or opposite weekly bias
            if close[i] <= highest_high[i] or close[i] >= weekly_pivot[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel or opposite weekly bias
            if close[i] >= lowest_low[i] or close[i] <= weekly_pivot[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0