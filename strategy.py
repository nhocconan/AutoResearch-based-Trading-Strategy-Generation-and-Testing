#!/usr/bin/env python3
"""
4h Bollinger Band Breakout with Volume Confirmation and Trend Filter
Hypothesis: Bollinger Band breakouts capture momentum moves, especially when 
confirmed by volume and aligned with higher timeframe trend (EMA200 on 1d).
Works in both bull and bear markets by filtering breakouts with trend direction.
Target: 20-40 trades/year (80-160 over 4 years) to avoid fee drag.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bb_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) - calculated on close
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    # Daily EMA200 for trend filter (no look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper[i]   # Break above upper BB
        breakdown_down = close[i] < bb_lower[i] # Break below lower BB
        
        # Trend filter: only take longs in uptrend, shorts in downtrend
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_filter and uptrend
        short_entry = breakdown_down and volume_filter and downtrend
        
        # Exit conditions: return to middle band
        long_exit = close[i] < bb_middle[i]
        short_exit = close[i] > bb_middle[i]
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals