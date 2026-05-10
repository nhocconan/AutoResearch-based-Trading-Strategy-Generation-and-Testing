#!/usr/bin/env python3
# 6h_Weekly_Pivot_Donchian_Breakout_Trend
# Hypothesis: Breakouts from weekly pivot zones (R1/S1) with 12h trend and volume confirmation.
# Weekly pivot levels act as strong support/resistance. Breakouts above R1 in uptrend or below S1 in downtrend
# capture momentum moves. Volume surge confirms breakout validity. Works in bull markets via buying breakouts
# and in bear markets via selling breakdowns. Targets 20-50 trades/year to minimize fee drag.

name = "6h_Weekly_Pivot_Donchian_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support and resistance levels
    s1_1w = 2 * pivot_1w - high_1w
    r1_1w = 2 * pivot_1w - low_1w
    s2_1w = pivot_1w - (high_1w - low_1w)
    r2_1w = pivot_1w + (high_1w - low_1w)
    
    # Align weekly levels to 6h timeframe (only update when new weekly bar is available)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA30 for trend filter
    close_12h = df_12h['close'].values
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) for breakout confirmation
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (1) + Donchian (20) + EMA (30)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_30_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h EMA
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > ema_30_12h_aligned[i]
        downtrend = close_12h_aligned[i] < ema_30_12h_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Price position relative to weekly pivot levels
        price_above_r1 = close[i] > r1_1w_aligned[i]
        price_below_s1 = close[i] < s1_1w_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_up = close[i] > highest_high[i]
        donchian_breakout_down = close[i] < lowest_low[i]
        
        if position == 0:
            # Long: price breaks above R1 pivot level with Donchian breakout, volume surge and 12h uptrend
            if price_above_r1 and donchian_breakout_up and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 pivot level with Donchian breakout, volume surge and 12h downtrend
            elif price_below_s1 and donchian_breakout_down and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below R1 pivot level OR trend changes
            if close[i] < r1_1w_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above S1 pivot level OR trend changes
            if close[i] > s1_1w_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals