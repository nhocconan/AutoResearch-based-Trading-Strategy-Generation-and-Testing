#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w EMA Filter and Volume Confirmation
Hypothesis: Donchian breakouts on daily timeframe capture major trends, filtered by 1-week EMA to avoid counter-trend trades, with volume confirmation to ensure breakout strength. Designed for 30-100 trades over 4 years to minimize fee drag while adapting to bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for EMA filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper and lower bands
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)  # For EMA and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or EMA filter violation
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR price below 1w EMA
            if close[i] <= donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR price above 1w EMA
            if close[i] >= donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA filter + volume
            bull_breakout = close[i] > donchian_upper[i]
            bear_breakout = close[i] < donchian_lower[i]
            
            # EMA filter: only long above EMA, short below EMA
            above_ema = close[i] > ema_50_1w_aligned[i]
            below_ema = close[i] < ema_50_1w_aligned[i]
            
            # Volume confirmation
            vol_filter = volume[i] > vol_ma[i] * 1.5
            
            if bull_breakout and above_ema and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and below_ema and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals