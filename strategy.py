#!/usr/bin/env python3
"""
12h_1w_Donchian_20_Trend_Filter_v1
Hypothesis: Price breaking above/below weekly Donchian(20) high/low on 12h timeframe, filtered by weekly EMA20 trend. Uses weekly timeframe for trend and structure to avoid noise, with 12h for execution. Trend filter ensures alignment with longer-term momentum. Designed to work in bull (uptrend breaks) and bear (downtrend breaks). Target: 15-30 trades/year to avoid fee drag.
"""

name = "12h_1w_Donchian_20_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # --- Weekly Trend Filter: EMA20 ---
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # --- Weekly Donchian Channels (20-period) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                # Simple trend-following exit: reverse when trend changes
                if position == 1 and close_12h[i] < ema20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] > ema20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine weekly trend
        trend_up = close_12h[i] > ema20_1w_aligned[i]
        trend_down = close_12h[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend
            if close_12h[i] > donchian_high_aligned[i] and trend_up:
                # Long: price breaks above weekly Donchian high + weekly uptrend
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < donchian_low_aligned[i] and trend_down:
                # Short: price breaks below weekly Donchian low + weekly downtrend
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Exit when price crosses the weekly EMA20 (trend change)
            if position == 1:
                if close_12h[i] < ema20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close_12h[i] > ema20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals