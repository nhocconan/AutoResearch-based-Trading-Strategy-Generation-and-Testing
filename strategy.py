#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter
# Donchian breakout provides clear structure with proven edge in trending and ranging markets
# Volume confirmation ensures breakout authenticity, reducing false signals
# 1d EMA50 filter aligns with higher timeframe trend to avoid counter-trend trades
# Discrete sizing 0.30 targets 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)

name = "4h_Donchian20_Volume_1dEMA50"
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
    
    # Calculate Donchian channels (20-period) using prior completed candle
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND volume spike AND 1d EMA50 uptrend
            if close[i] > highest_high[i] and volume[i] > (2.0 * vol_ema_20[i]) and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower band AND volume spike AND 1d EMA50 downtrend
            elif close[i] < lowest_low[i] and volume[i] > (2.0 * vol_ema_20[i]) and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle (or lower band for tighter stop)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Donchian middle (or upper band for tighter stop)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals