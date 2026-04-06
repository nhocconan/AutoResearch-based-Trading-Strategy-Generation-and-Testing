#!/usr/bin/env python3
"""
12h Donchian Breakout with Daily Trend and Volume Confirmation
Hypothesis: On 12h timeframe, Donchian(20) breakouts aligned with daily EMA(50) trend and volume confirmation capture significant moves while avoiding false breakouts. Works in bull (long with uptrend) and bear (short with downtrend) via symmetric logic. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_ema_volume_v6"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend and Donchian calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian channels
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (low[i] <= low_roll[i] or
                close[i] <= entry_price - 2.5 * (high_roll[i] - low_roll[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (high[i] >= high_roll[i] or
                close[i] >= entry_price + 2.5 * (high_roll[i] - low_roll[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + daily trend alignment + volume
            long_breakout = high[i] > high_roll[i]
            short_breakout = low[i] < low_roll[i]
            
            uptrend = ema_50_1d_aligned[i] > close_1d[-1] if i == len(ema_50_1d_aligned)-1 else ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            downtrend = ema_50_1d_aligned[i] < close_1d[-1] if i == len(ema_50_1d_aligned)-1 else ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
            
            vol_filter = volume[i] > (1.5 * vol_ma[i])
            
            if long_breakout and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals