#!/usr/bin/env python3
# 1D_WeeklyTrend_Donchian20_Breakout
# Hypothesis: Use weekly Donchian channel breakouts with daily trend filter to capture major moves.
# Long when price breaks above weekly Donchian high (20-period) and daily close > daily EMA50.
# Short when price breaks below weekly Donchian low (20-period) and daily close < daily EMA50.
# Weekly trend acts as primary filter, reducing whipsaws in ranging markets.
# Target: 15-25 trades/year per symbol, suitable for 1d timeframe.

name = "1D_WeeklyTrend_Donchian20_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly Donchian channel (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian bands on weekly data
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian bands to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Daily trend filter: close > EMA50 for uptrend, < EMA50 for downtrend
    daily_uptrend = close > ema50
    daily_downtrend = close < ema50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50[i]) or np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: weekly breakout above Donchian high + daily uptrend
            if close[i] > high_20_aligned[i] and daily_uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly breakout below Donchian low + daily downtrend
            elif close[i] < low_20_aligned[i] and daily_downtrend[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly Donchian mid-point or trend reverses
            mid_point = (high_20_aligned[i] + low_20_aligned[i]) / 2
            if close[i] < mid_point or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly Donchian mid-point or trend reverses
            mid_point = (high_20_aligned[i] + low_20_aligned[i]) / 2
            if close[i] > mid_point or not daily_downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals