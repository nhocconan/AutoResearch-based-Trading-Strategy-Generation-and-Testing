#!/usr/bin/env python3
"""
1h_KAMA_Trend_Filter_4hHigherHighsLowerLows
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) to filter trend on 1h, combined with 4h higher highs/lows for structural trend confirmation. Only trades in direction of higher timeframe structure to avoid counter-trend whipsaws. Designed for low trade frequency (15-30/year) with high win rate in both bull and bear markets by avoiding false breakouts. KAMA adapts to market noise, reducing false signals during sideways periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA on 1h (trend filter)
    # Efficiency ratio: |price change| / sum of absolute price changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    for i in range(10, n):  # 10-period ER
        if np.sum(abs_change[i-9:i+1]) > 0:
            er[i] = change[i] / np.sum(abs_change[i-9:i+1])
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 4h data for structure (higher highs/lows)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h swing highs/lows (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Find swing highs (local maxima)
    swing_high = np.zeros(len(high_4h), dtype=bool)
    swing_low = np.zeros(len(low_4h), dtype=bool)
    for i in range(10, len(high_4h)-10):
        if high_4h[i] == np.max(high_4h[i-10:i+11]):
            swing_high[i] = True
        if low_4h[i] == np.min(low_4h[i-10:i+11]):
            swing_low[i] = True
    
    # Get most recent swing high/low values
    last_swing_high = np.full(len(high_4h), np.nan)
    last_swing_low = np.full(len(high_4h), np.nan)
    last_high_val = np.nan
    last_low_val = np.nan
    for i in range(len(high_4h)):
        if swing_high[i]:
            last_high_val = high_4h[i]
        if swing_low[i]:
            last_low_val = low_4h[i]
        last_swing_high[i] = last_high_val
        last_swing_low[i] = last_low_val
    
    # Align swing levels to 1h
    swing_high_aligned = align_htf_to_ltf(prices, df_4h, last_swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_4h, last_swing_low)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for KAMA to stabilize
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(swing_high_aligned[i]) or
            np.isnan(swing_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # Structure filter: 4h higher highs/lows
        # Uptrend: price above recent swing high AND making higher highs
        # Downtrend: price below recent swing low AND making lower lows
        uptrend_structure = close[i] > swing_high_aligned[i] and (i < 2 or close[i] > close[i-1])
        downtrend_structure = close[i] < swing_low_aligned[i] and (i < 2 or close[i] < close[i-1])
        
        # Entry conditions
        long_entry = above_kama and uptrend_structure
        short_entry = below_kama and downtrend_structure
        
        # Exit conditions: opposite structure break
        long_exit = close[i] < swing_low_aligned[i]  # Break of structure
        short_exit = close[i] > swing_high_aligned[i]  # Break of structure
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_KAMA_Trend_Filter_4hHigherHighsLowerLows"
timeframe = "1h"
leverage = 1.0