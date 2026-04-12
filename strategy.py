#!/usr/bin/env python3
"""
4h_12h_Camarilla_Triangle_Breakout
Hypothesis: Combine 12h trend (EMA21) with 4h breakout from contracting triangle pattern.
Go long when price breaks above triangle resistance in uptrend, short when breaks below support in downtrend.
Triangle identified by higher lows and lower highs over 5 periods. Volume confirmation required.
Designed to capture breakouts with momentum in trending markets, avoiding false breakouts in ranges.
Target: 60-120 total trades over 4 years (15-30/year) on 4h timeframe.
Works in bull (breakouts continuation) and bear (breakdowns continuation).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Triangle_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # === VOLUME FILTER (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === TRIANGLE DETECTION (5-period higher lows & lower highs) ===
    # Higher lows: each low > previous low
    hl_condition = np.ones(n, dtype=bool)
    for i in range(1, 5):
        hl_condition[i:] &= (low[i:] > low[:-i])
    
    # Lower highs: each high < previous high
    hh_condition = np.ones(n, dtype=bool)
    for i in range(1, 5):
        hh_condition[i:] &= (high[i:] < high[:-i])
    
    triangle_condition = hl_condition & hh_condition
    
    # Triangle boundaries: recent high and low
    resistance = pd.Series(high).rolling(window=5, min_periods=5).max().values
    support = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema21_12h_aligned[i]) or np.isnan(resistance[i]) or 
            np.isnan(support[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from 12h EMA21
        close_12h_arr = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h_arr)
        trend_up = close_12h_aligned[i] > ema21_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema21_12h_aligned[i]
        
        # Breakout conditions: price breaks triangle boundary + trend + volume
        breakout_up = close[i] > resistance[i-1] and triangle_condition[i-1]
        breakout_down = close[i] < support[i-1] and triangle_condition[i-1]
        
        long_signal = breakout_up and trend_up and vol_ratio[i] > 1.5
        short_signal = breakout_down and trend_down and vol_ratio[i] > 1.5
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = (position == 1 and 
                    (breakout_down or not trend_up))
        exit_short = (position == -1 and 
                     (breakout_up or not trend_down))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals