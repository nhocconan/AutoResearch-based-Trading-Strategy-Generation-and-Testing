#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_Confirmation
Hypothesis: 6h Donchian(20) breakouts confirmed by weekly trend direction (weekly EMA50).
Breakouts above upper band in weekly uptrend go long; breakdowns below lower band in weekly downtrend go short.
Weekly trend filter reduces false breakouts in sideways markets, improving win rate during BTC/ETH bear/range phases.
Target: 50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on 6h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for weekly EMA and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_roll[i]
        lower = low_roll[i]
        weekly_trend_up = price > ema_50_1w_aligned[i]  # Proxy: price above weekly EMA50 = uptrend
        weekly_trend_down = price < ema_50_1w_aligned[i]  # Price below weekly EMA50 = downtrend
        
        if position == 0:
            # Long: break above upper band in weekly uptrend
            if price > upper and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band in weekly downtrend
            elif price < lower and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below upper band OR weekly trend turns down
            if price < upper or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above lower band OR weekly trend turns up
            if price > lower or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_Confirmation"
timeframe = "6h"
leverage = 1.0