#!/usr/bin/env python3
"""
1h_Pivot_Breakout_4hTrend_Volume
Hypothesis: Breakouts from hourly pivot points (R1/S1) confirmed by 4h trend and volume spikes.
Works in bull/bear markets via 4h trend filter - only takes longs in uptrend, shorts in downtrend.
Volume filter prevents false breakouts in low liquidity. Session filter (08-20 UTC) reduces noise.
Targets 15-37 trades/year by requiring confluence of pivot break, trend, and volume.
"""

name = "1h_Pivot_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h average volume for volume filter
    vol_avg_4h = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    # Calculate hourly pivot points from previous hour OHLC
    # Standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Pre-calculate session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA50 (50) and 4h vol avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_4h_aligned[i]) or 
            np.isnan(pivot[i]) or 
            np.isnan(r1[i]) or 
            np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (4h)
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: current 1h volume > 1.5x average 4h volume
        vol_1h = volume[i]
        volume_filter = vol_1h > vol_avg_4h_aligned[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 resistance + uptrend + volume
            if close[i] > r1[i] and uptrend_4h and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 support + downtrend + volume
            elif close[i] < s1[i] and downtrend_4h and volume_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot or trend breaks
            if close[i] < pivot[i] or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to pivot or trend breaks
            if close[i] > pivot[i] or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals