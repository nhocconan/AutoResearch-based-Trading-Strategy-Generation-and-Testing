#!/usr/bin/env python3
"""
6h Elder Ray with 1d trend filter and volume confirmation
Hypothesis: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure.
In bull markets: buy when Bull Power > 0 and EMA13 rising; in bear markets: sell when Bear Power < 0 and EMA13 falling.
1d EMA50 filters trend direction to avoid counter-trend trades. Volume confirms institutional participation.
Works in both bull (buy strength) and bear (sell weakness). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA13 and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema13[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: Bear Power turns negative OR stoploss
            if (bear_power[i] < 0 or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power turns positive OR stoploss
            if (bull_power[i] > 0 or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray + trend + volume
            bull_entry = (bull_power[i] > 0 and 
                         ema50_rising_aligned[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            bear_entry = (bear_power[i] < 0 and 
                         ema50_falling_aligned[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals