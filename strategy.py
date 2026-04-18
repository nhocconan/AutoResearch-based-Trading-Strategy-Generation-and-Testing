#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend Filter
Long: Close breaks above Donchian(20) high + weekly EMA200 rising
Short: Close breaks below Donchian(20) low + weekly EMA200 falling
Exit: Opposite break or price crosses Donchian midpoint
Designed to capture trend continuations in both bull and bear markets.
Target: 30-100 total trades over 4 years (7-25/year)
"""

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20) channels
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = period20_high.values
    donchian_low = period20_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to daily
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate weekly EMA200 slope (1-period change) for trend filter
    ema200_slope = np.diff(ema200_1w_aligned, prepend=ema200_1w_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian calculations
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(ema200_slope[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: break above Donchian high + rising weekly EMA200
            if price > donchian_high[i] and ema200_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + falling weekly EMA200
            elif price < donchian_low[i] and ema200_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low OR price crosses below midpoint
            if price < donchian_low[i] or price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high OR price crosses above midpoint
            if price > donchian_high[i] or price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0