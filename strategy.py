#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_TrendFilter_v1
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction and EMA50 trend capture high-probability moves in both bull and bear markets.
Weekly pivot (based on prior week OHLC) provides institutional reference levels; EMA50 filter avoids counter-trend trades.
Discrete sizing (0.25) balances return and fee drag. Target: 75-150 total trades over 4 years.
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
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: PP = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1 = 2*PP - L, S1 = 2*PP - H
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h Donchian(20) breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Donchian (20), weekly EMA50 (50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        pp = pp_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        ema50 = ema50_1w_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs weekly EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend:
                # Long bias: long when price breaks above Donchian high AND above weekly R1 (bullish pivot)
                if close_val > upper and close_val > r1:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend:
                # Short bias: short when price breaks below Donchian low AND below weekly S1 (bearish pivot)
                if close_val < lower and close_val < s1:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: stop at Donchian low or weekly S1 touch
            if close_val < lower or close_val < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: stop at Donchian high or weekly R1 touch
            if close_val > upper or close_val > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0