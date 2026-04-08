#!/usr/bin/env python3
"""
4h_1d_1w_pivot_breakout_volume_v3
Hypothesis: Use 4h price action with 1d pivot levels and 1w trend bias.
Long when 4h price breaks above 1d R1 with 1w bullish trend and volume confirmation.
Short when 4h price breaks below 1d S1 with 1w bearish trend and volume confirmation.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) by requiring strong breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_pivot_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d pivot points (standard floor trader pivots)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d pivot point
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    # 1d resistance and support levels
    r1 = 2 * pivot_point - low_1d
    s1 = 2 * pivot_point - high_1d
    r2 = pivot_point + (high_1d - low_1d)
    s2 = pivot_point - (high_1d - low_1d)
    
    # Align 1d pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1w trend bias: close > EMA(50) for bullish, close < EMA(50) for bearish
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish = close_1w > ema_50
    trend_bearish = close_1w < ema_50
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish.astype(float))
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(trend_bullish_aligned[i]) or
            np.isnan(trend_bearish_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 1d S1 or 1w trend turns bearish
            if close[i] < s1_aligned[i] or trend_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 1d R1 or 1w trend turns bullish
            if close[i] > r1_aligned[i] or trend_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 1d R1 with 1w bullish trend and volume
            if close[i] > r1_aligned[i] and trend_bullish_aligned[i] > 0.5 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 1d S1 with 1w bearish trend and volume
            elif close[i] < s1_aligned[i] and trend_bearish_aligned[i] > 0.5 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals