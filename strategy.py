#!/usr/bin/env python3
"""
6h Elder Ray with Weekly Trend Filter and Volume Confirmation
Hypothesis: Elder Ray (bull/bear power) identifies institutional buying/selling pressure.
In bull markets: buy when bull power > 0 and EMA13 > 0. In bear markets: short when bear power < 0 and EMA13 < 0.
Weekly trend filter ensures we trade with the dominant trend. Volume confirms institutional participation.
Works in both bull and bear by switching between long and short based on weekly trend.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_weekly_trend_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(20) for long-term trend
    close_weekly = df_weekly['close'].values
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # 6h EMA(13) - Elder Ray uses EMA13 as reference
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull power: high - EMA13
    bear_power = low - ema13   # Bear power: low - EMA13
    
    # 6h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require above-average volume
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA13 and ATR14
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema20_weekly_aligned[i]) or np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = ema20_weekly_aligned[i] > 0  # Price above weekly EMA20
        weekly_downtrend = ema20_weekly_aligned[i] < 0  # Price below weekly EMA20
        
        # Check exits
        if position == 1:  # long position
            # Exit: bear power turns negative OR stoploss
            if (bear_power[i] < 0 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turns positive OR stoploss
            if (bull_power[i] > 0 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray + weekly trend + volume
            long_setup = (bull_power[i] > 0 and weekly_uptrend and vol_filter[i])
            short_setup = (bear_power[i] < 0 and weekly_downtrend and vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals