#!/usr/bin/env python3
# 1D_WeeklyTrend_Donchian20_Breakout
# Hypothesis: Buy breakouts above 20-day Donchian high in weekly uptrend, sell breakdowns below 20-day Donchian low in weekly downtrend.
# Uses weekly trend filter to avoid counter-trend trades. Works in bull/bear by following the weekly trend.
# Target: 15-25 trades/year per symbol.

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
    volume = prices['volume'].values
    
    # Daily indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Donchian channels (20-period)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above Donchian high
            if weekly_up and close[i] > donchian_high[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price breaks below Donchian low
            elif weekly_down and close[i] < donchian_low[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down OR price breaks below Donchian low
            if not weekly_up or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up OR price breaks above Donchian high
            if not weekly_down or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals