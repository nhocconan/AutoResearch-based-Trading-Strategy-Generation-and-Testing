#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Regime_v1
Hypothesis: Daily Camarilla breakout with 1d regime filter (chop > 61.8 = range, chop < 38.2 = trend)
Enter long when price breaks above daily R4 AND chop < 38.2 (trending up).
Enter short when price breaks below daily S4 AND chop < 38.2 (trending down).
Exit when price returns to daily midpoint or chop > 61.8 (range).
Uses 12h primary timeframe for execution, targeting 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for regime and levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r4 = prev_close + range_ * 1.1 / 2
    camarilla_s4 = prev_close - range_ * 1.1 / 2
    camarilla_midpoint = prev_close
    
    # Handle invalid ranges
    valid_range = range_ > 0
    camarilla_r4 = np.where(valid_range, camarilla_r4, np.nan)
    camarilla_s4 = np.where(valid_range, camarilla_s4, np.nan)
    camarilla_midpoint = np.where(valid_range, camarilla_midpoint, np.nan)
    
    # Chop index for regime detection (14-period)
    hl_range = df_1d['high'] - df_1d['low']
    atr14 = hl_range.rolling(window=14, min_periods=14).mean()
    sum_abs_change = ((df_1d['close'] - df_1d['close'].shift(1)).abs()).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(sum_abs_change / atr14 / 14) / np.log10(2)
    chop_values = chop.values
    
    # Align to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_midpoint_aligned = align_htf_to_ltf(prices, df_1d, camarilla_midpoint)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_midpoint_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: chop < 38.2 = trending, chop > 61.8 = ranging
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        # Breakout conditions (only in trending regime)
        long_breakout = is_trending and high[i] > camarilla_r4_aligned[i]
        short_breakout = is_trending and low[i] < camarilla_s4_aligned[i]
        
        # Exit conditions: return to midpoint OR entering ranging regime
        long_exit = (not is_trending) or close[i] < camarilla_midpoint_aligned[i]
        short_exit = (not is_trending) or close[i] > camarilla_midpoint_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals