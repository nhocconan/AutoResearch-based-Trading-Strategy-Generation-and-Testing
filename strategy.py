#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Enter long when price breaks above Donchian(20) upper band in uptrend (1d EMA50 rising).
# Enter short when price breaks below Donchian(20) lower band in downtrend (1d EMA50 falling).
# Volume > 1.5x 20-period average confirms breakout strength.
# Exit on opposite Donchian breakout or when price crosses 1d EMA50.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_donchian20_1dema50_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR crosses below EMA50
            if close[i] < lowest_low_20[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR crosses above EMA50
            if close[i] > highest_high_20[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA50 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_20[i] and close[i] > ema_50_aligned[i]:
                    # Breakout above upper band in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and close[i] < ema_50_aligned[i]:
                    # Breakdown below lower band in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals