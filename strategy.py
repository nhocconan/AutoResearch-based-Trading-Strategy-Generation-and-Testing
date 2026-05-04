#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian breakouts capture strong momentum moves. 1d EMA50 ensures alignment with higher timeframe trend.
# Volume spike confirms institutional participation. Targets 12-37 trades/year on 6f timeframe.
# Works in bull markets via breakout continuation and bear markets via short breakdowns with trend filter.

name = "6h_Donchian20_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian(20) channels on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper band AND price > 1d EMA50 AND volume spike
            if close[i] > highest_high[i] and close[i] > ema50_1d_aligned[i] and volume[i] > (2.0 * np.nanmedian(volume[max(0, i-20):i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower band AND price < 1d EMA50 AND volume spike
            elif close[i] < lowest_low[i] and close[i] < ema50_1d_aligned[i] and volume[i] > (2.0 * np.nanmedian(volume[max(0, i-20):i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band OR below 1d EMA50
            if close[i] < lowest_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band OR above 1d EMA50
            if close[i] > highest_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals