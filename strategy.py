#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter
# Donchian breakouts capture breakout momentum. Volume confirms institutional participation.
# 12h EMA filter ensures we only trade in the direction of the intermediate trend.
# This combination works in both bull and bear markets by filtering for trend-aligned breakouts.
# Targets 20-50 trades per year (~80-200 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_Volume_12hEMA"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20-1, n):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h
    ema_50 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_50[i] = (close_12h[i] - ema_50[i-1]) * multiplier + ema_50[i-1]
    
    # Trend: close > EMA50 = uptrend, close < EMA50 = downtrend
    trend_up = close_12h > ema_50
    trend_down = close_12h < ema_50
    
    # Align to 4h timeframe
    trend_up_4h = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_4h = align_htf_to_ltf(prices, df_12h, trend_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback-1, 20-1)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i]) or np.isnan(trend_up_4h[i]) or 
            np.isnan(trend_down_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, volume confirms, uptrend
            if close[i] > highest_high[i] and vol_confirm[i] and trend_up_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, volume confirms, downtrend
            elif close[i] < lowest_low[i] and vol_confirm[i] and trend_down_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian or trend turns down
            if close[i] < lowest_low[i] or trend_down_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper Donchian or trend turns up
            if close[i] > highest_high[i] or trend_up_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals