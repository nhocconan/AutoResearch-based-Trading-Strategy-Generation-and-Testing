#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour timeframe with 4-hour EMA(20) trend filter and 1-day Donchian(15) breakout.
# Uses 4h EMA for medium-term trend direction and 1d Donchian breakout for entry signals.
# Entry timing on 1h with volume confirmation to reduce false breakouts.
# Volume > 1.5x 20-period average confirms breakout strength.
# Exit when price reverses through opposite Donchian level or crosses 4h EMA.
# Session filter (08-20 UTC) to avoid low-liquidity periods.
# Target: 80-150 total trades over 4 years (20-38/year) to balance signal quality and fee drag.

name = "1h_1d_donchian15_4hema20_vol_v1"
timeframe = "1h"
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    # 4h EMA(20) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # 1-day Donchian(15) channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    highest_high_15 = pd.Series(high_1d).rolling(window=15, min_periods=15).max().values
    lowest_low_15 = pd.Series(low_1d).rolling(window=15, min_periods=15).min().values
    highest_high_15_aligned = align_htf_to_ltf(prices, df_1d, highest_high_15)
    lowest_low_15_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_15)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_aligned[i]) or np.isnan(highest_high_15_aligned[i]) or 
            np.isnan(lowest_low_15_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below 1d Donchian lower OR crosses below 4h EMA20
            if close[i] < lowest_low_15_aligned[i] or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above 1d Donchian upper OR crosses above 4h EMA20
            if close[i] > highest_high_15_aligned[i] or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: 1d Donchian breakout + 4h EMA20 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_15_aligned[i] and close[i] > ema_20_aligned[i]:
                    # Breakout above 1d Donchian upper in uptrend: long
                    signals[i] = 0.20
                    position = 1
                elif close[i] < lowest_low_15_aligned[i] and close[i] < ema_20_aligned[i]:
                    # Breakdown below 1d Donchian lower in downtrend: short
                    signals[i] = -0.20
                    position = -1
    
    return signals