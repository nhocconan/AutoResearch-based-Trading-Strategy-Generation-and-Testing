#!/usr/bin/env python3
"""
12h_1w_Donchian_20_WeekTrend_V1
Hypothesis: 12h Donchian(20) breakout in direction of weekly trend (price > weekly EMA50) provides high-probability trend continuation. Weekly filter reduces whipsaw in choppy markets. Works in bull/bear by only taking trades aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian upper and lower bands
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(len(high)):
        start = max(0, i - 19)
        donchian_high[i] = np.max(high[start:i+1])
        donchian_low[i] = np.min(low[start:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        weekly_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + price above weekly EMA50
            if price > upper and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + price below weekly EMA50
            elif price < lower and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below Donchian low or below weekly EMA50
            if price < lower or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above Donchian high or above weekly EMA50
            if price > upper or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Donchian_20_WeekTrend_V1"
timeframe = "12h"
leverage = 1.0