#!/usr/bin/env python3
"""
6H_ELDER_RAY_BULL_POWER_BEAR_POWER_1D_TREND_FILTER
Hypothesis: Elder Ray Index (bull/bear power) with 1-day trend filter captures
institutional buying/selling pressure in both bull and bear markets. 
Uses 1-day EMA13 trend filter to align with higher timeframe direction,
reducing false signals during counter-trend moves. Designed for ~15-30 trades/year
on 6h to minimize fee drag while capturing meaningful momentum shifts.
"""
name = "6H_ELDER_RAY_BULL_POWER_BEAR_POWER_1D_TREND_FILTER"
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
    
    # Calculate EMA13 for Elder Ray (13-period EMA standard)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter (standard for trend identification)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA13 warmup
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (buying pressure) AND price above 1d EMA34 (uptrend)
            if bull_power[i] > 0 and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (selling pressure) AND price below 1d EMA34 (downtrend)
            elif bear_power[i] < 0 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power becomes negative (selling pressure appears)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power becomes positive (buying pressure appears)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals