#!/usr/bin/env python3
# 1d_Donchian_Breakout_1wTrend_VolumeFilter
# Hypothesis: On daily timeframe, enter long when price breaks above Donchian(20) high with weekly EMA40 uptrend and volume > 1.5x 20-day average.
# Enter short when price breaks below Donchian(20) low with weekly EMA40 downtrend and volume > 1.5x 20-day average.
# Exit when price crosses back below Donchian(20) midpoint for longs or above midpoint for shorts.
# Uses weekly trend filter to avoid counter-trend trades and volume confirmation to avoid false breakouts.
# Targets 15-25 trades/year for low fee drag and works in both bull and bear markets by following the weekly trend.

name = "1d_Donchian_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-day)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Weekly EMA40 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema40 = pd.Series(weekly_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    weekly_ema40_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema40)
    
    # Volume confirmation: 20-day moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_ema40_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with weekly uptrend and volume > 1.5x MA
            if (close[i] > highest_high[i] and 
                close[i] > weekly_ema40_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with weekly downtrend and volume > 1.5x MA
            elif (close[i] < lowest_low[i] and 
                  close[i] < weekly_ema40_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals