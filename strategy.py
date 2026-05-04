#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian breakouts capture strong momentum moves, EMA50 filters for higher timeframe trend alignment,
# volume spike confirms institutional participation. Works in both bull and bear markets due to
# trend filter preventing counter-trend entries and volume confirmation reducing false breakouts.
# Targets 20-40 trades/year to minimize fee drag while maintaining edge.

name = "4h_Donchian20_1dEMA50_VolumeSpike"
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
    
    # Calculate Donchian channels (20-period) on 4h data
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper AND 1d EMA50 uptrend AND volume spike
            if close[i] > highest_high[i] and close[i] > ema50_1d_aligned[i] and volume[i] > (2.0 * np.nanmedian(volume[max(0, i-50):i])):
                signals[i] = 0.30
                position = 1
            # Short conditions: break below Donchian lower AND 1d EMA50 downtrend AND volume spike
            elif close[i] < lowest_low[i] and close[i] < ema50_1d_aligned[i] and volume[i] > (2.0 * np.nanmedian(volume[max(0, i-50):i])):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR below 1d EMA50
            if close[i] < lowest_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Donchian upper OR above 1d EMA50
            if close[i] > highest_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals