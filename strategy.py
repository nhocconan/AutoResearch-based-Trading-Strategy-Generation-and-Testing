#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Volume Confirmation and ATR Stop
Hypothesis: Donchian channel breakouts capture strong momentum moves.
In bull markets: break above 20-period high signals continuation.
In bear markets: break below 20-period low signals continuation.
1d volume filter ensures institutional participation. Works in both regimes.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for volume filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume EMA20 for confirmation
    vol_1d = df_1d['volume'].values
    vol_ema_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_1d_prev = np.roll(vol_ema_1d, 1)
    vol_ema_1d_prev[0] = vol_ema_1d[0]
    vol_increasing = vol_ema_1d > vol_ema_1d_prev
    vol_ema_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_1d)
    vol_increasing_aligned = align_htf_to_ltf(prices, df_1d, vol_increasing)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # ATR for stoploss (14-period)
    def calculate_atr(high_arr, low_arr, close_arr, period):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = np.full_like(tr, np.nan)
        for i in range(period, len(tr)):
            if i == period:
                atr[i] = np.mean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14)  # For Donchian and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ema_1d_aligned[i]) or 
            np.isnan(vol_increasing_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: break below lower Donchian OR stoploss
            if (close[i] <= donchian_low[i] or 
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: break above upper Donchian OR stoploss
            if (close[i] >= donchian_high[i] or 
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume confirmation
            long_entry = (close[i] > donchian_high[i] and 
                         vol_increasing_aligned[i])
            short_entry = (close[i] < donchian_low[i] and 
                          vol_increasing_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals