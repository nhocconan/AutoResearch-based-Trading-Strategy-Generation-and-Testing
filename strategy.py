#!/usr/bin/env python3
name = "1d_WeeklyTrend_Filter_Donchian"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 21:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Donchian channel (20-day)
    # Upper band: 20-day high
    # Lower band: 20-day low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 days for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_21_1d[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only take longs in weekly uptrend, shorts in weekly downtrend
        weekly_uptrend = close[i] > ema_21_1d[i]
        weekly_downtrend = close[i] < ema_21_1d[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band + weekly uptrend
            if close[i] > donchian_upper[i-1] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band + weekly downtrend
            elif close[i] < donchian_lower[i-1] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below Donchian lower band OR weekly trend reversal
            if close[i] < donchian_lower[i-1] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above Donchian upper band OR weekly trend reversal
            if close[i] > donchian_upper[i-1] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals