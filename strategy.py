#!/usr/bin/env python3
"""
4h_WeeklyPivot_Donchian_Breakout_Trend_Volume
Hypothesis: Uses weekly pivot levels (weekly high/low) as structural support/resistance combined with Donchian channel breakouts for trend confirmation. 
In bull markets: buy when price breaks above weekly high + Donchian(20) breakout + volume confirmation.
In bear markets: sell when price breaks below weekly low + Donchian(20) breakdown + volume confirmation.
Uses 1d trend filter (EMA50) for higher timeframe bias. Weekly pivot provides institutional levels that work in both bull/bear regimes.
Target: 20-40 trades/year per symbol.
"""

name = "4h_WeeklyPivot_Donchian_Breakout_Trend_Volume"
timeframe = "4h"
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
    
    # Weekly pivot levels: weekly high and low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_ratio = volume / vol_ma
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if volume data not available
        if np.isnan(volume_ratio[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        if position == 0:
            # LONG: price > weekly high AND Donchian breakout AND volume spike AND 1d uptrend
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > highest_high[i] and 
                volume_ratio[i] > 1.5 and 
                uptrend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price < weekly low AND Donchian breakdown AND volume spike AND 1d downtrend
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < lowest_low[i] and 
                  volume_ratio[i] > 1.5 and 
                  downtrend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < weekly low OR Donchian breakdown
            if close[i] < weekly_low_aligned[i] or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > weekly high OR Donchian breakout
            if close[i] > weekly_high_aligned[i] or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals