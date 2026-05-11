#!/usr/bin/env python3
name = "6h_WeeklyPivot_DailyTrend_Continuation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Load weekly data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 2. Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 3. Weekly pivot points (classic formula: P = (H+L+C)/3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    # R2 = P + (H-L), S2 = P - (H-L)
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # 4. Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # 5. Align daily EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6. Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 150
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_pivot = close[i] > pivot_1w_aligned[i]
        price_below_pivot = close[i] < pivot_1w_aligned[i]
        price_above_ema1d = close[i] > ema50_1d_aligned[i]
        price_below_ema1d = close[i] < ema50_1d_aligned[i]
        breakout_r2 = close[i] > r2_1w_aligned[i]
        breakdown_s2 = close[i] < s2_1w_aligned[i]
        pullback_r1 = close[i] < r1_1w_aligned[i] and close[i] > pivot_1w_aligned[i]
        pullback_s1 = close[i] > s1_1w_aligned[i] and close[i] < pivot_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly R2 with daily uptrend + volume
            if breakout_r2 and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below weekly S2 with daily downtrend + volume
            elif breakdown_s2 and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size
                position = -1
            # Long pullback: Price pulls back to R1 in uptrend
            elif pullback_r1 and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short pullback: Price pulls back to S1 in downtrend
            elif pullback_s1 and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price breaks below weekly S1 OR trend reverses
                if close[i] < s1_1w_aligned[i] or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks above weekly R1 OR trend reverses
                if close[i] > r1_1w_aligned[i] or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals