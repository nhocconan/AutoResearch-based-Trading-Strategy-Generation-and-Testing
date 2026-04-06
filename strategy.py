#!/usr/bin/env python3
"""
4h Donchian breakout with 12h volume confirmation and ATR stoploss
Hypothesis: Donchian(20) breakouts capture trend continuation in both bull and bear markets.
12h volume > 1.5x EMA confirms institutional participation. ATR-based stoploss limits drawdown.
Designed for 4h timeframe to achieve 75-200 total trades over 4 years with controlled frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for volume confirmation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h volume EMA for confirmation
    vol_12h = df_12h['volume'].values
    vol_ema_12h = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_12h_ema_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_12h)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_12h_ema_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss (2*ATR)
            if (close[i] <= low_min[i] or 
                close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss (2*ATR)
            if (close[i] >= high_max[i] or 
                close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume confirmation
            long_breakout = close[i] > high_max[i]
            short_breakout = close[i] < low_min[i]
            volume_ok = volume[i] > vol_12h_ema_aligned[i] * 1.5
            
            if long_breakout and volume_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and volume_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals