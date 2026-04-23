#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and 1d weekly pivot confirmation
- Long: Price breaks above Donchian(20) high + price > 4h EMA50 + price > 1d weekly pivot R1
- Short: Price breaks below Donchian(20) low + price < 4h EMA50 + price < 1d weekly pivot S1
- Exit: Price crosses Donchian midpoint (mean reversion to avoid whipsaws)
- Uses 4h EMA50 for trend direction, 1d weekly pivot for institutional levels
- Volume confirmation removed to reduce trade frequency (target: 15-35 trades/year)
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Position size: 0.20 (discrete to minimize fee churn)
- Designed for 1h timeframe to work in both bull and bear markets by combining breakout with trend and institutional filters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = (2 * pivot) - weekly_low  # Resistance 1
    s1 = (2 * pivot) - weekly_high  # Support 1
    
    # Align HTF pivot levels to LTF
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ms], so we can use DatetimeIndex properties
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 5)  # Donchian(20), EMA50, weekly pivot(5)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above prior period high
        breakout_down = close[i] < lowest_low[i-1]   # Break below prior period low
        
        # Trend filter: price relative to 4h EMA50
        above_ema = close[i] > ema_50_4h_aligned[i]
        below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Weekly pivot filter
        above_r1 = close[i] > r1_aligned[i]
        below_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up + above 4h EMA50 + above weekly R1
            if breakout_up and above_ema and above_r1:
                signals[i] = 0.20
                position = 1
            # Short: Donchian breakout down + below 4h EMA50 + below weekly S1
            elif breakout_down and below_ema and below_s1:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Price crosses Donchian midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian midpoint
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price rises above Donchian midpoint
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_4hEMA50_1dWeeklyPivot_R1S1_Breakout"
timeframe = "1h"
leverage = 1.0