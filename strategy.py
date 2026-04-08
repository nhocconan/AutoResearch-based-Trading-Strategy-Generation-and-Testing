#!/usr/bin/env python3
# 6h_weekly_pivot_breakout
# Hypothesis: Breakout strategy using weekly pivot levels (from 1w data) with 6h price action and volume confirmation.
# In both bull and bear markets, price tends to respect weekly support/resistance levels.
# Breakouts above weekly R3 or below S3 with volume indicate strong momentum.
# Uses 12h trend filter to avoid counter-trend trades.
# Target: 15-30 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

name = "6h_weekly_pivot_breakout"
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
    
    # Weekly data for pivot points (1w) - load once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pp_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pp_1w)
    
    # Align weekly pivots to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # 12h trend filter - load once before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 6h indicators
    # EMA20 for dynamic reference
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema20[i]) or np.isnan(avg_volume[i]) or np.isnan(ema50_12h_aligned[i]) or \
           np.isnan(pp_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # 12h trend filter
        trend_up = close[i] > ema50_12h_aligned[i]
        trend_down = close[i] < ema50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: close below EMA20 or reversal below weekly pivot
            if close[i] < ema20[i] or close[i] < pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above EMA20 or reversal above weekly pivot
            if close[i] > ema20[i] or close[i] > pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long breakout: price crosses above weekly R3 in uptrend
                if trend_up and close[i] > r3_1w_aligned[i] and close[i-1] <= r3_1w_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price crosses below weekly S3 in downtrend
                elif trend_down and close[i] < s3_1w_aligned[i] and close[i-1] >= s3_1w_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals