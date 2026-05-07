#!/usr/bin/env python3
name = "1d_Weekly_Pivot_Crossover_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_ltf_to_htf(prices, df_1w, weekly_pivot)
    r1_aligned = align_ltf_to_htf(prices, df_1w, weekly_r1)
    s1_aligned = align_ltf_to_htf(prices, df_1w, weekly_s1)
    r2_aligned = align_ltf_to_htf(prices, df_1w, weekly_r2)
    s2_aligned = align_ltf_to_htf(prices, df_1w, weekly_s2)
    
    # Daily volume filter: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above S1 with volume (bullish bounce from support)
            if close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below R1 with volume (bearish rejection from resistance)
            elif close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below S2 (strong support break) or above R2 (overbought)
            if close[i] < s2_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above R2 (strong resistance break) or below S2 (oversold)
            if close[i] > r2_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points provide significant support/resistance levels
# that work in both bull and bear markets. Price respecting these levels
# with volume confirmation creates high-probability entries. The strategy
# buys near weekly support (S1) and sells near weekly resistance (R1),
# with exits at stronger levels (S2/R2) to capture meaningful moves.
# Using weekly pivots on daily timeframe reduces noise and avoids overtrading.
# Position size 0.25 limits risk while allowing meaningful returns. 
# Target: 15-25 trades/year to minimize fee drag in choppy 2025 market.