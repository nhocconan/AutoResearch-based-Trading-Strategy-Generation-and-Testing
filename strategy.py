#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with volume confirmation and daily EMA trend filter.
# Uses 4h timeframe to balance trade frequency and capture medium-term moves.
# Donchian(20) breakouts capture breakout momentum, volume confirms institutional interest.
# Daily EMA100 filter ensures alignment with higher timeframe trend for multi-timeframe confluence.
# Designed to work in both bull and bear markets by following daily trend direction.
name = "4h_Donchian20_1dEMA100_Volume_Trend"
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
    
    # Daily data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period) - calculated on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA100 trend filter
    ema_100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_4h = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Volume spike filter: volume > 1.8x 20-period EMA (higher threshold to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_100_4h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with volume spike and above daily EMA100
            if (price > high_20[i] and vol_spike[i] and price > ema_100_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with volume spike and below daily EMA100
            elif (price < low_20[i] and vol_spike[i] and price < ema_100_4h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-period low (mean reversion to support)
            if price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-period high (mean reversion to resistance)
            if price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals