#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(15) breakout with weekly EMA(20) trend filter and volume confirmation.
# Enter long when price breaks above Donchian(15) upper band in uptrend (weekly EMA20 rising).
# Enter short when price breaks below Donchian(15) lower band in downtrend (weekly EMA20 falling).
# Volume > 1.8x 15-period average confirms breakout strength.
# Exit on opposite Donchian breakout or when price crosses weekly EMA20.
# Target: 50-150 total trades over 4 years (12-37/year) to balance signal quality and fee drag.

name = "12h_donchian15_weekly_ema20_vol_v1"
timeframe = "12h"
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
    
    # Weekly EMA(20) for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    ema_20 = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_weekly, ema_20)
    
    # Donchian(15) channels
    highest_high_15 = pd.Series(high).rolling(window=15, min_periods=15).max().values
    lowest_low_15 = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # Volume confirmation: volume > 1.8x 15-period average
    volume_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_threshold = 1.8 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(15, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR crosses below EMA20
            if close[i] < lowest_low_15[i] or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR crosses above EMA20
            if close[i] > highest_high_15[i] or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA20 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_15[i] and close[i] > ema_20_aligned[i]:
                    # Breakout above upper band in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_15[i] and close[i] < ema_20_aligned[i]:
                    # Breakdown below lower band in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals