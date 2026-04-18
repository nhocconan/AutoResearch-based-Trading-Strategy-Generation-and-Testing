#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_With_Daily_Trend_v2
Hypothesis: Price breaking above/below weekly pivot R4/S4 with daily trend filter and volume confirmation captures strong momentum moves in both bull and bear markets. Weekly pivots act as strong support/resistance levels, and breakouts with volume indicate institutional participation. Daily trend filter ensures trades align with higher timeframe momentum. Target: 20-40 trades/year (80-160 total over 4 years).
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
    
    # Weekly high/low/close for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot levels: R4, S4
    # R4 = Close + 3*(High - Low)
    # S4 = Close - 3*(High - Low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    r4 = weekly_close + 3 * (weekly_high - weekly_low)
    s4 = weekly_close - 3 * (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1w, r4)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4)
    
    # Daily EMA trend filter (34-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for volume MA and pivot alignment
    
    for i in range(start_idx, n):
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema_1d_6h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r4_val = r4_6h[i]
        s4_val = s4_6h[i]
        vol_ok = volume_filter[i]
        daily_trend = ema_1d_6h[i]
        
        if position == 0:
            # Long: price breaks above weekly R4 with volume in uptrend
            if price > r4_val and vol_ok and price > daily_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 with volume in downtrend
            elif price < s4_val and vol_ok and price < daily_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price breaks below weekly S4 or trend reverses
            if price < s4_6h[i] or price < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price breaks above weekly R4 or trend reverses
            if price > r4_6h[i] or price > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Breakout_With_Daily_Trend_v2"
timeframe = "6h"
leverage = 1.0