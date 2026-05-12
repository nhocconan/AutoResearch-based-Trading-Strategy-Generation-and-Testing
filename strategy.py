#!/usr/bin/env python3
# 6h_Donchian_Breakout_WeeklyTrend_VolumeConfirmation
# Hypothesis: Use 6h Donchian channel breakouts with weekly trend filter and volume confirmation.
# Long when price breaks above 20-period high with price > weekly EMA and volume > 1.5x 20-period average.
# Short when price breaks below 20-period low with price < weekly EMA and volume > 1.5x 20-period average.
# Exit when price reverses back into the Donchian channel (crosses middle band).
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures breakouts have institutional participation.
# Designed to work in both bull and bear markets by filtering with weekly trend.
# Targets 15-25 trades/year to minimize fee drag.

name = "6h_Donchian_Breakout_WeeklyTrend_VolumeConfirmation"
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
    
    # Calculate Donchian Channel (20-period)
    highest = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = (highest + lowest) / 2
    
    # Weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high with price > weekly EMA and volume > 1.5x MA
            if close[i] > highest[i] and close[i] > weekly_ema_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low with price < weekly EMA and volume > 1.5x MA
            elif close[i] < lowest[i] and close[i] < weekly_ema_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back below middle band
            if close[i] < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back above middle band
            if close[i] > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals