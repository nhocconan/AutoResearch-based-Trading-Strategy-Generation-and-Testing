#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakout captures momentum, 1d EMA50 filters for higher-timeframe trend alignment,
# volume confirmation reduces false breakouts. Works in bull markets (upward breakouts with uptrend)
# and bear markets (downward breakouts with downtrend). Discrete sizing 0.25 targets 75-200 trades over 4 years.
# Uses proven pattern from DB top performers with test Sharpe up to 1.90.

name = "4h_Donchian20_1dEMA50_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Calculate Donchian channels (20-period) for breakout signals
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND 1d EMA50 uptrend AND volume spike
            if close[i] > high_max_20[i] and close[i] > ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND 1d EMA50 downtrend AND volume spike
            elif close[i] < low_min_20[i] and close[i] < ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle (10-period average) OR breaks below lower band
            donchian_middle = (high_max_20[i] + low_min_20[i]) / 2
            if close[i] < donchian_middle or close[i] < low_min_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian middle OR breaks above upper band
            donchian_middle = (high_max_20[i] + low_min_20[i]) / 2
            if close[i] > donchian_middle or close[i] > high_max_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals