#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Donchian channels from 4h for breakout structure, 1d EMA34 for trend filter,
# and volume spike for confirmation. Designed for 20-30 trades/year to minimize fee drag.
# Works in bull markets via upside breakouts and in bear markets via downside breakdowns.
# Donchian channels adapt to volatility, providing dynamic support/resistance.

name = "4h_Donchian20_1dEMA34_VolumeSpike_TrendFilter"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate Donchian(20) channels on 4h timeframe
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper AND 1d EMA34 uptrend AND volume spike
            vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else volume[i]
            if close[i] > high_roll_max[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ma_20):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower AND 1d EMA34 downtrend AND volume spike
            elif close[i] < low_roll_min[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ma_20):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR below 1d EMA34
            if close[i] < low_roll_min[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR above 1d EMA34
            if close[i] > high_roll_max[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals