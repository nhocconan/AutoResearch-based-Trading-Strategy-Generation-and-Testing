#!/usr/bin/env python3
"""
12h_1w_Donchian_Breakout_With_Trend_Filter
Hypothesis: Use weekly Donchian(20) breakouts with 1d trend filter (close > EMA50) and volume confirmation.
Long when price breaks above weekly upper band with volume > 1.5x 24-bar avg AND close > EMA50.
Short when price breaks below weekly lower band with volume > 1.5x 24-bar avg AND close < EMA50.
Exit when price crosses back through weekly middle band (mean of upper/lower).
Designed for 12h timeframe to capture multi-week moves with ~15-35 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data once for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian(20) channels
    upper = np.full_like(high_1w, np.nan)
    lower = np.full_like(low_1w, np.nan)
    middle = np.full_like(close_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        upper[i] = np.max(high_1w[i-20:i])
        lower[i] = np.min(low_1w[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Shift to align with current week (channels based on past 20 weeks)
    upper = np.roll(upper, 1)
    lower = np.roll(lower, 1)
    middle = np.roll(middle, 1)
    upper[0] = np.nan
    lower[0] = np.nan
    middle[0] = np.nan
    
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    middle_aligned = align_htf_to_ltf(prices, df_1w, middle)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: 24-period average (2 days of 12h data)
    vol_ma = prices['volume'].rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long conditions: break above upper band + volume confirmation + price above EMA50
            if price > upper_aligned[i] and volume_ok and price > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + volume confirmation + price below EMA50
            elif price < lower_aligned[i] and volume_ok and price < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below middle band
            if price < middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above middle band
            if price > middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Donchian_Breakout_With_Trend_Filter"
timeframe = "12h"
leverage = 1.0