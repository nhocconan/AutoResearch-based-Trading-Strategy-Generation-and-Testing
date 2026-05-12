#!/usr/bin/env python3
"""
1d_Donchian_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
Enters long when price breaks above 20-day high with weekly uptrend and volume above average.
Enters short when price breaks below 20-day low with weekly downtrend and volume above average.
Uses weekly timeframe for trend filter to avoid counter-trend trades.
Designed for low trade frequency (15-25/year) to minimize fee drag on BTC/ETH.
"""

name = "1d_Donchian_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Calculate Donchian channels (20-day high/low)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-day average volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get current values
        donchian_high = period20_high[i]
        donchian_low = period20_low[i]
        ema50_aligned = ema50_1w_aligned[i]
        vol_ma = vol_ma20[i]
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high) or np.isnan(donchian_low) or 
            np.isnan(ema50_aligned) or np.isnan(vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 20-day high + weekly uptrend + volume above average
            if (close[i] > donchian_high and 
                close[i] > ema50_aligned and 
                volume[i] > vol_ma):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-day low + weekly downtrend + volume above average
            elif (close[i] < donchian_low and 
                  close[i] < ema50_aligned and 
                  volume[i] > vol_ma):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-day low (reverse signal) or trend changes
            if close[i] < donchian_low or close[i] < ema50_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-day high (reverse signal) or trend changes
            if close[i] > donchian_high or close[i] > ema50_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals