#!/usr/bin/env python3

# 1d_weekly_pivot_breakout_v1
# Hypothesis: Weekly pivot point breakout with daily trend filter and volume confirmation.
# Uses weekly pivot levels (R1, R2, S1, S2) as entry levels. Enters on breakout with
# volume confirmation when daily price is above/below EMA200. Exits when price returns
# to weekly pivot point. Designed to work in both bull and bear markets by capturing
# breakouts from weekly consolidation zones.
# Target: 15-25 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_breakout_v1"
timeframe = "1d"
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
    
    # Weekly data for pivot points (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2.0 * pivot - low_1w
    s1 = 2.0 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily trend filter: EMA200
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: 20-day average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Need EMA200 warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema200[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or \
           np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema200[i]
        daily_downtrend = close[i] < ema200[i]
        
        if position == 1:  # Long position
            # Exit when price returns to weekly pivot point
            if close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price returns to weekly pivot point
            if close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long breakout: price crosses above R1 in uptrend
                if daily_uptrend and close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price crosses below S1 in downtrend
                elif daily_downtrend and close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals