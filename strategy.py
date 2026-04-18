#!/usr/bin/env python3
"""
1d_ThreeDay_Low_High_Breakout_Volume_Trend
Hypothesis: Breakouts above the 3-day high or below the 3-day low on daily timeframe with volume confirmation and weekly trend filter capture momentum moves. Works in bull markets via upward breakouts and bear markets via downward breakdowns. Weekly trend filter ensures alignment with higher timeframe momentum to reduce whipsaws. Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 3-day high and low for breakout levels
    high_3d = pd.Series(high).rolling(window=3, min_periods=3).max().values
    low_3d = pd.Series(low).rolling(window=3, min_periods=3).min().values
    
    # Volume filter: >1.5x 10-day average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Weekly trend filter: EMA21 on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 10  # Warmup for volume MA and 3-day high/low
    
    for i in range(start_idx, n):
        if (np.isnan(high_3d[i]) or np.isnan(low_3d[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_level = high_3d[i]
        low_level = low_3d[i]
        vol_ok = volume_filter[i]
        weekly_trend = ema_21_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above 3-day high with volume in bullish weekly trend
            if price > high_level and vol_ok and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 3-day low with volume in bearish weekly trend
            elif price < low_level and vol_ok and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to 3-day low or weekly trend turns bearish
            if price < low_3d[i] or price < weekly_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to 3-day high or weekly trend turns bullish
            if price > high_3d[i] or price > weekly_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_ThreeDay_Low_High_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0