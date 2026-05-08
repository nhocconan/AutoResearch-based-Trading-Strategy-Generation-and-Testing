#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 4h volume confirmation.
# Long when price breaks above Donchian upper channel AND 4h volume > 2.0x 20-period average AND price > 12h EMA50.
# Short when price breaks below Donchian lower channel AND 4h volume > 2.0x 20-period average AND price < 12h EMA50.
# Exit when price crosses below/above Donchian middle line (20-period average) to capture mean reversion.
# Uses Donchian for trend capture with volume confirmation to avoid false breakouts.
# Target: 80-160 total trades over 4 years (20-40/year) for low fee drift.

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # 4h volume filter: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, volume spike, above 12h EMA50
            long_cond = (close[i] > donchian_upper[i]) and (close[i-1] <= donchian_upper[i-1]) and volume_filter[i] and (close[i] > ema50_12h_aligned[i])
            # Short conditions: price breaks below Donchian lower, volume spike, below 12h EMA50
            short_cond = (close[i] < donchian_lower[i]) and (close[i-1] >= donchian_lower[i-1]) and volume_filter[i] and (close[i] < ema50_12h_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian middle (mean reversion signal)
            if close[i] < donchian_middle[i] and close[i-1] >= donchian_middle[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian middle (mean reversion signal)
            if close[i] > donchian_middle[i] and close[i-1] <= donchian_middle[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals