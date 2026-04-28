#!/usr/bin/env python3
"""
12h_Pivot_Trend_Rebound
Hypothesis: Price often reverses from intraday extremes toward the daily pivot point in ranging markets.
We use the daily Pivot Point (PP) as a mean-reversion target, filtered by the 12h trend (EMA50) to avoid
counter-trend trades. Enter long when price is below PP in an uptrend, short when above PP in a downtrend.
Exit when price crosses back over PP. This strategy targets low-frequency mean reversion in both
bull and bear markets by aligning with the higher timeframe trend. Uses 12h timeframe to keep trades
within 50-150 per year, minimizing fee drag. The pivot acts as a dynamic equilibrium level.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get daily data for Pivot Point calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily Pivot Point: (H + L + C) / 3
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(pp_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Mean reversion signals around daily pivot
        long_signal = close[i] < pp_aligned[i] and uptrend
        short_signal = close[i] > pp_aligned[i] and downtrend
        
        # Exit when price crosses back over pivot
        long_exit = close[i] > pp_aligned[i]
        short_exit = close[i] < pp_aligned[i]
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
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
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Pivot_Trend_Rebound"
timeframe = "12h"
leverage = 1.0