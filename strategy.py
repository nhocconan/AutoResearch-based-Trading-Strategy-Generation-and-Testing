#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA50) and volume confirmation (>2.0x 20 EMA volume)
# Uses Donchian channels from prior completed 6h bar for structure (breakout of 20-period high/low)
# Weekly EMA50 filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaw
# Volume confirmation ensures breakout has strong participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 80-160 total trades over 4 years = 20-40/year for 6h timeframe
# This strategy focuses on strong breakouts with trend alignment, which should work in both bull (long bias) and bear (short bias) markets.

name = "6h_Donchian20_WeeklyEMA50_VolumeSpike"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    # Calculate weekly EMA50 trend filter from prior completed weekly bar
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Shift by 1 to use only prior completed weekly bar (no look-ahead)
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) from prior completed 6h bar
    # We need to calculate this manually since we can't use future data
    high_20 = np.full_like(high, np.nan)
    low_20 = np.full_like(low, np.nan)
    
    for i in range(20, len(high)):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    # Shift by 1 to use only prior completed 6h bar (no look-ahead)
    high_20_shifted = np.roll(high_20, 1)
    low_20_shifted = np.roll(low_20, 1)
    high_20_shifted[0] = np.nan
    low_20_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(high_20_shifted[i]) or np.isnan(low_20_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-period high + price > weekly EMA50 + volume spike
            if close[i] > high_20_shifted[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 20-period low + price < weekly EMA50 + volume spike
            elif close[i] < low_20_shifted[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel OR weekly trend filter fails
            midpoint = (high_20_shifted[i] + low_20_shifted[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel OR weekly trend filter fails
            midpoint = (high_20_shifted[i] + low_20_shifted[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals