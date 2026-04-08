#!/usr/bin/env python3
# 6h_1d_weekly_pivot_trend_v1
# Hypothesis: Use weekly pivot point levels to determine trend direction, with daily confirmation and 6h entry timing.
# In bull markets: price stays above weekly pivot (bullish bias), look for longs on dips to support.
# In bear markets: price stays below weekly pivot (bearish bias), look for shorts on rallies to resistance.
# Weekly pivot provides structural support/resistance that works in both regimes.
# Daily timeframe filters out noise, 6h provides timely entries.
# Target: 20-50 trades/year to avoid fee drag while capturing meaningful swings.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot points (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Get daily data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_20 = np.zeros(len(close_1d))
    ema_1d_20[:] = np.nan
    if len(close_1d) >= 20:
        ema_1d_20[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_1d_20[i] = close_1d[i] * 0.0952 + ema_1d_20[i-1] * 0.9048
    
    ema_1d_20_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure we have enough data
    start_idx = max(20, len(prices) - len(prices))  # Just ensure we start after warmup
    for i in range(20, n):
        price = close[i]
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        r2 = weekly_r2_aligned[i]
        s2 = weekly_s2_aligned[i]
        ema_1d = ema_1d_20_aligned[i]
        
        # Skip if any values are NaN
        if (np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or 
            np.isnan(r2) or np.isnan(s2) or np.isnan(ema_1d)):
            if position != 0:
                pass  # Hold current position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below weekly S1 OR daily EMA20 turns down
            if price < s1 or price < ema_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above weekly R1 OR daily EMA20 turns up
            if price > r1 or price > ema_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat - look for entry
            # Long entry: price above weekly pivot AND above daily EMA20 (bullish alignment)
            # Enter near support (weekly S1 or S2) on pullbacks
            if price > pivot and price > ema_1d:
                # Look for pullback to weekly support levels
                if price <= s1 * 1.005:  # Within 0.5% of S1
                    position = 1
                    signals[i] = 0.25
                elif price <= s2 * 1.005:  # Within 0.5% of S2 (deeper pullback)
                    position = 1
                    signals[i] = 0.25
                    
            # Short entry: price below weekly pivot AND below daily EMA20 (bearish alignment)
            # Enter near resistance (weekly R1 or R2) on bounces
            elif price < pivot and price < ema_1d:
                # Look for bounce to weekly resistance levels
                if price >= r1 * 0.995:  # Within 0.5% of R1
                    position = -1
                    signals[i] = -0.25
                elif price >= r2 * 0.995:  # Within 0.5% of R2 (stronger bounce)
                    position = -1
                    signals[i] = -0.25
    
    return signals