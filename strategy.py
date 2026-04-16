# [EXPERIMENT #51095] Hypothesis: 6h timeframe with weekly pivot-based trend filter and daily volatility-adjusted breakout.
# Uses weekly pivot direction (from prior week) to filter breakouts on 6h chart.
# Only takes long when price > weekly pivot and breaks above daily resistance;
# only short when price < weekly pivot and breaks below daily support.
# Volatility filter: requires ATR(14) expansion to avoid chop.
# Target: 12-37 trades/year (50-150 over 4 years) with size 0.25.
# Weekly pivot provides structural bias; daily breakout provides timing; volatility filter reduces false signals.

#!/usr/bin/env python3
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
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate ATR on 6h
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_avg = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
    atr_6h_avg_aligned = align_htf_to_ltf(prices, df_6h, atr_6h_avg)
    
    # === 1d data (for daily support/resistance) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily resistance/support (pivot-based)
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    # Prepend first value to maintain length
    pivot_1d = np.concatenate([[pivot_1d[0]], pivot_1d])
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily levels to 6h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1w data (for weekly trend bias) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot (from prior week)
    weekly_pivot = (high_1w[:-1] + low_1w[:-1] + close_1w[:-1]) / 3.0
    weekly_pivot = np.concatenate([[weekly_pivot[0]], weekly_pivot])
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_avg_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        atr_avg = atr_6h_avg_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        daily_pivot = pivot_1d_aligned[i]
        daily_r1 = r1_1d_aligned[i]
        daily_s1 = s1_1d_aligned[i]
        
        # Volatility filter: require ATR expansion (> average)
        vol_expansion = atr_6h[i] > atr_avg if not np.isnan(atr_6h[i]) else False
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below daily pivot OR weekly bias flips
            if (price < daily_pivot) or (price < weekly_pivot_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above daily pivot OR weekly bias flips
            if (price > daily_pivot) or (price > weekly_pivot_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > weekly pivot (bullish bias) AND breaks above daily R1 AND volatility expansion
            if (price > weekly_pivot_val) and (price > daily_r1) and vol_expansion:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price < weekly pivot (bearish bias) AND breaks below daily S1 AND volatility expansion
            elif (price < weekly_pivot_val) and (price < daily_s1) and vol_expansion:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Bias_DailyBreakout_VolFilter"
timeframe = "6h"
leverage = 1.0