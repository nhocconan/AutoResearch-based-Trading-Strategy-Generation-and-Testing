#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Donchian breakout captures momentum, 1w EMA50 filters for higher timeframe trend,
# volume confirmation ensures breakout validity. Works in bull markets (long breakouts
# with uptrend) and bear markets (short breakouts with downtrend).
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
# Discrete sizing 0.25 to minimize fee churn.

name = "6h_Donchian20_1wEMA50_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter from prior completed 1w bar
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_shifted = np.roll(ema50_1w, 1)
    ema50_1w_shifted[0] = np.nan
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_shifted)
    
    # Calculate Donchian channels (20-period) for breakout
    # Upper channel: highest high over past 20 periods
    # Lower channel: lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 1w EMA50 uptrend AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 1w EMA50 downtrend AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR 1w EMA50 turns down
            if close[i] < donchian_lower[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR 1w EMA50 turns up
            if close[i] > donchian_upper[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals