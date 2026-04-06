#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly EMA Filter and Volume Confirmation v1
Hypothesis: Daily Donchian(20) breakouts capture strong trends; weekly EMA200 filters counter-trend trades to avoid whipsaws in ranging markets; volume confirms breakout strength. Designed for 30-100 trades over 4 years to minimize fee drag while adapting to bull/bear markets via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_weekly_ema200_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA200
    weekly_close = df_weekly['close'].values
    weekly_ema200 = pd.Series(weekly_close).ewm(span=200, min_periods=200, adjust=False).mean().values
    weekly_ema200_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema200)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(200, 20)  # For weekly EMA200 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_ema200_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or weekly EMA reversal
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR weekly close crosses below EMA200
            if low[i] < donchian_low[i] or close[i] < weekly_ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR weekly close crosses above EMA200
            if high[i] > donchian_high[i] or close[i] > weekly_ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly EMA filter + volume
            bull_breakout = high[i] > donchian_high[i]
            bear_breakout = low[i] < donchian_low[i]
            weekly_uptrend = close[i] > weekly_ema200_aligned[i]
            weekly_downtrend = close[i] < weekly_ema200_aligned[i]
            volume_ok = volume[i] > vol_ma[i] * 1.5
            
            if bull_breakout and weekly_uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and weekly_downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals