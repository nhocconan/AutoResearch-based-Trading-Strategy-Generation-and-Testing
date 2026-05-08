#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d EMA50 trend filter and 12h volume confirmation.
# Long when price breaks above Donchian high (20) AND price > 1d EMA50 AND 12h volume > 1.5x 20-period average.
# Short when price breaks below Donchian low (20) AND price < 1d EMA50 AND 12h volume > 1.5x 20-period average.
# Exit when price crosses back below Donchian midline (for long) or above midline (for short).
# Uses Donchian for breakout in ranging/trending markets, EMA50 for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume spike, above 1d EMA50
            long_cond = (close[i] > highest_high[i]) and volume_filter[i] and (close[i] > ema50_1d_aligned[i])
            # Short conditions: price breaks below Donchian low, volume spike, below 1d EMA50
            short_cond = (close[i] < lowest_low[i]) and volume_filter[i] and (close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian midline
            if close[i] < donchian_mid[i] and close[i-1] >= donchian_mid[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian midline
            if close[i] > donchian_mid[i] and close[i-1] <= donchian_mid[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals