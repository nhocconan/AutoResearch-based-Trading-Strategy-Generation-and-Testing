#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_12hTrend_Volume
Hypothesis: Weekly pivot levels from weekly data provide robust support/resistance. Price breaking above R1 with 12h EMA20 uptrend and volume spike captures bullish moves; breaking below S1 with 12h EMA20 downtrend and volume spike captures bearish moves. Uses 6h as primary timeframe to capture multi-day moves while minimizing trade frequency. Weekly pivots adapt to long-term structure, working in both bull and bear markets by following the dominant weekly trend filtered by 12h EMA and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Standard pivot: P = (H + L + C) / 3
    pivot = (high_w + low_w + close_w) / 3
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    r1_weekly = 2 * pivot - low_w
    s1_weekly = 2 * pivot - high_w
    
    # Align weekly levels to 6h timeframe (use prior week's levels)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or 
            np.isnan(ema20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = r1_weekly_aligned[i]
        s1 = s1_weekly_aligned[i]
        ema_trend = ema20_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and volume spike
            if close[i] > r1 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with downtrend and volume spike
            elif close[i] < s1 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below S1 or trend turns down
            if close[i] < s1 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above R1 or trend turns up
            if close[i] > r1 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Direction_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0