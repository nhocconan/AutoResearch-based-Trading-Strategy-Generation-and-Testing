#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_v2
# Hypothesis: Breakout of weekly pivot levels (R1/S1, R2/S2) with 12h trend filter and volume confirmation. Works in bull/bear by aligning with weekly structure and filtering by 12h trend. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation (lookback 5 days)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2.0 * pivot_1w - low_1w
    s1_1w = 2.0 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or \
           np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # 12h trend filter
        trend_up = close[i] > ema50_12h_aligned[i]
        trend_down = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price falls below S1 or trend reverses
            if close[i] < s1_1w_aligned[i] or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above R1 or trend reverses
            if close[i] > r1_1w_aligned[i] or not trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long breakout: price closes above R1 with upward trend
                if trend_up and close[i] > r1_1w_aligned[i] and close[i-1] <= r1_1w_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below S1 with downward trend
                elif trend_down and close[i] < s1_1w_aligned[i] and close[i-1] >= s1_1w_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals