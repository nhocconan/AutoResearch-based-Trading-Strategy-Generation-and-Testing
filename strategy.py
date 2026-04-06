#!/usr/bin/env python3
"""
1d Donchian breakout with volume confirmation and 1w trend filter
Hypothesis: Donchian(20) breakouts on 1d with volume > 1.5x 20-day average and 
1w EMA50 trend filter capture institutional breakouts while avoiding counter-trend trades.
Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_prev = np.roll(ema50_1w, 1)
    ema50_1w_prev[0] = ema50_1w[0]
    ema50_rising = ema50_1w > ema50_1w_prev
    ema50_falling = ema50_1w < ema50_1w_prev
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_falling)
    
    # 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= low_20[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= high_20[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > high_20[i]
            bear_breakout = close[i] < low_20[i]
            vol_filter = volume[i] > vol_ema[i] * 1.5
            
            if bull_breakout and ema50_rising_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and ema50_falling_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals