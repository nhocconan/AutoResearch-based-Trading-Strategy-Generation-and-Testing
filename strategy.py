#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1d weekly pivot (PP), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, price < 1d weekly pivot (PP), volume > 1.5x avg
# Exit when: price retraces to midpoint of Donchian channel OR opposite breakout occurs
# Uses daily pivot to define trend (above PP = bullish, below PP = bearish) and avoids counter-trend trades
# Weekly pivot provides more stable trend filter than daily EMA, reducing whipsaws in sideways markets
# Target: 50-150 trades over 4 years (12-37/year) with size 0.25

name = "6h_donchian20_1dweeklypivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 1d weekly pivot point (PP) for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Weekly pivot: PP = (high + low + close) / 3 (using previous day's data)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + pivot trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > pp_1d_aligned[i]:
                    # Bullish breakout above Donchian high with price above weekly pivot
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and close[i] < pp_1d_aligned[i]:
                    # Bearish breakdown below Donchian low with price below weekly pivot
                    signals[i] = -0.25
                    position = -1
    
    return signals