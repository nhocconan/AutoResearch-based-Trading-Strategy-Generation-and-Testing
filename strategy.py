#!/usr/bin/env python3
# 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
# Hypothesis: Donchian breakouts capture momentum, weekly pivot provides directional bias
# from higher timeframe structure, and volume confirmation filters false breakouts.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Designed for low trade frequency (~15-25/year) to avoid fee drag.
name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
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
    
    # === Weekly Data for Pivot Direction ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly Pivot Points (standard calculation)
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Weekly trend: price above/below pivot
    weekly_trend = np.where(weekly_close > pivot, 1, -1)  # 1=uptrend, -1=downtrend
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # === Donchian Channel (20-period) on 6h ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Volume Confirmation (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)  # 1.5x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + weekly uptrend + volume spike
            if (close[i] > highest_high[i] and 
                weekly_trend_aligned[i] > 0 and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + weekly downtrend + volume spike
            elif (close[i] < lowest_low[i] and 
                  weekly_trend_aligned[i] < 0 and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (breakout failed)
            if close[i] < highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel (breakdown failed)
            if close[i] > lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals