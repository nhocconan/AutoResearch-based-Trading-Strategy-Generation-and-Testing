#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h EMA trend filter
# Donchian breakouts capture momentum in trending markets. Volume confirmation ensures
# institutional participation. EMA filter avoids false breakouts in weak trends.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for strong trends only.

name = "4h_Donchian20_12hVolume_12hEMA"
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
    
    # Donchian channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for volume and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Volume confirmation on 12h (volume > 1.5x 20-period EMA)
    vol_ema = pd.Series(df_12h['volume']).ewm(span=20, adjust=False).mean().values
    vol_threshold = vol_ema * 1.5
    vol_confirmed = df_12h['volume'].values > vol_threshold
    vol_confirmed_4h = align_htf_to_ltf(prices, df_12h, vol_confirmed)
    
    # EMA trend filter on 12h (price > EMA50 for long, price < EMA50 for short)
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False).mean().values
    ema_trend_up = df_12h['close'].values > ema_50
    ema_trend_down = df_12h['close'].values < ema_50
    ema_trend_up_4h = align_htf_to_ltf(prices, df_12h, ema_trend_up)
    ema_trend_down_4h = align_htf_to_ltf(prices, df_12h, ema_trend_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirmed_4h[i]) or np.isnan(ema_trend_up_4h[i]) or 
            np.isnan(ema_trend_down_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume confirmed, uptrend
            if close[i] > highest_high[i] and vol_confirmed_4h[i] and ema_trend_up_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume confirmed, downtrend
            elif close[i] < lowest_low[i] and vol_confirmed_4h[i] and ema_trend_down_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend reverses
            if close[i] < lowest_low[i] or not ema_trend_up_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or trend reverses
            if close[i] > highest_high[i] or not ema_trend_down_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals