#!/usr/bin/env python3
# 4h_Donchian20_Volume_Trend_Signal
# Hypothesis: Donchian channel breakouts capture breakout momentum. Combined with volume confirmation and trend filter from 1-day EMA34 to enter in direction of higher timeframe trend.
# Works in bull markets (long on upper band breakout + uptrend) and bear markets (short on lower band breakout + downtrend).
# Volume spike filters low-conviction moves. Target: 20-40 trades/year per symbol.

name = "4h_Donchian20_Volume_Trend_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 2.0x average volume (40-period = ~3.33 days on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 40)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band, price above EMA34 (uptrend), volume spike
            if (high[i] > high_20[i-1] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band, price below EMA34 (downtrend), volume spike
            elif (low[i] < low_20[i-1] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below lower Donchian band OR price crosses below EMA34
            if (low[i] < low_20[i-1] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above upper Donchian band OR price crosses above EMA34
            if (high[i] > high_20[i-1] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals