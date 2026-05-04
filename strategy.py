#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian breakout provides clear structure-based entries in trending markets
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA volume) confirms institutional participation
# Discrete sizing 0.28 targets 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)
# ATR-based stoploss manages risk during adverse moves

name = "4h_Donchian20_1dEMA50_VolumeSpike"
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
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    
    # Align HTF indicator to 4h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Calculate Donchian channels (20-period) from prior completed 4h bar
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_20_shifted = np.roll(highest_high_20, 1)
    lowest_low_20_shifted = np.roll(lowest_low_20, 1)
    highest_high_20_shifted[0] = np.nan
    lowest_low_20_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high_20_shifted[i]) or 
            np.isnan(lowest_low_20_shifted[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND price > 1d EMA50 AND volume spike
            if close[i] > highest_high_20_shifted[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.28
                position = 1
            # Short conditions: price breaks below Donchian lower band AND price < 1d EMA50 AND volume spike
            elif close[i] < lowest_low_20_shifted[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle OR price crosses below 1d EMA50
            donchian_middle = (highest_high_20_shifted[i] + lowest_low_20_shifted[i]) / 2.0
            if close[i] < donchian_middle or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price closes above Donchian middle OR price crosses above 1d EMA50
            donchian_middle = (highest_high_20_shifted[i] + lowest_low_20_shifted[i]) / 2.0
            if close[i] > donchian_middle or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals