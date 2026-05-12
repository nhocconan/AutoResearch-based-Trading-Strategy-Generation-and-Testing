#!/usr/bin/env python3
# 1D_Donchian20_Breakout_WeeklyTrend_StockVolume
# Hypothesis: On daily timeframe, enter long when price breaks above 20-day Donchian high with weekly trend up and volume > 2x 20-day average.
# Enter short when price breaks below 20-day Donchian low with weekly trend down and volume > 2x 20-day average.
# Exit on opposite Donchian breakout or trend reversal.
# Weekly trend uses 200-day EMA proxy from weekly data to avoid whipsaws.
# Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull and bear markets.

name = "1D_Donchian20_Breakout_WeeklyTrend_StockVolume"
timeframe = "1d"
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
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: EMA200 proxy from weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema200 = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema200)
    
    # Volume confirmation: 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure weekly EMA200 is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or \
           np.isnan(weekly_ema200_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter: price vs weekly EMA200
        weekly_trend_up = close[i] > weekly_ema200_aligned[i]
        weekly_trend_down = close[i] < weekly_ema200_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high + weekly trend up + volume > 2x MA
            if close[i] > high_roll[i] and weekly_trend_up and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + weekly trend down + volume > 2x MA
            elif close[i] < low_roll[i] and weekly_trend_down and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR weekly trend turns down
            if close[i] < low_roll[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR weekly trend turns up
            if close[i] > high_roll[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals