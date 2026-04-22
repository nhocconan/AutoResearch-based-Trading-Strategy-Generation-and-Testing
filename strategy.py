#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Strategy: Weekly Pivot Breakout with Volume Confirmation on 6H
    Hypothesis: Weekly pivot levels act as strong support/resistance.
                Breakouts with volume confirmation capture momentum.
                Works in both bull/bear by trading breakouts in either direction.
                Uses weekly pivot from prior week, aligned to 6H.
                Target: 12-37 trades/year (50-150 over 4 years).
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points (ONCE before loop)
    df_wk = get_htf_data(prices, '1w')
    
    if len(df_wk) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    high_wk = df_wk['high'].values
    low_wk = df_wk['low'].values
    close_wk = df_wk['close'].values
    
    # Weekly pivot points (standard calculation)
    prev_high = high_wk
    prev_low = low_wk
    prev_close = close_wk
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    r2 = pivot + (prev_high - prev_low)
    s1 = 2 * pivot - prev_high
    s2 = pivot - (prev_high - prev_low)
    
    # Align weekly pivot levels to 6H timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_wk, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_wk, r1)
    r2_aligned = align_htf_to_ltf(prices, df_wk, r2)
    s1_aligned = align_htf_to_ltf(prices, df_wk, s1)
    s2_aligned = align_htf_to_ltf(prices, df_wk, s2)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R2 with volume spike
            if close[i] > r2_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2 with volume spike
            elif close[i] < s2_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite pivot level (full exit)
            if position == 1:
                # Exit long: Price closes below S1
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above R1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyPivot_R2_S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0