#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume spike confirmation
# Donchian breakouts capture strong momentum moves, filtered by weekly trend to avoid counter-trend trades
# Volume spike confirms institutional participation. Works in both bull and bear markets due to trend filter.
# Discrete sizing 0.25 targets 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Donchian20_1wEMA21_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1w data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA21 trend filter from prior completed 1w bar
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_shifted = np.roll(ema21_1w, 1)
    ema21_1w_shifted[0] = np.nan
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w_shifted)
    
    # Calculate Donchian(20) channels from prior completed 1d bar
    # We need 20 periods of high/low to calculate upper/lower bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper AND 1w EMA21 uptrend AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema21_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower AND 1w EMA21 downtrend AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema21_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR below 1w EMA21
            if close[i] < donchian_lower[i] or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR above 1w EMA21
            if close[i] > donchian_upper[i] or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals