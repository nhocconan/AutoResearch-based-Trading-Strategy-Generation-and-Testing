#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal with Volume Spike
# Uses weekly pivot points (S1, R1, S2, R2) from prior week. 
# Enters long when price touches S1/S2 with volume spike (>2x median) and RSI < 30 (oversold).
# Enters short when price touches R1/R2 with volume spike and RSI > 70 (overbought).
# Weekly pivot provides institutional support/resistance levels that work in both bull and bear markets.
# Volume spike confirms institutional interest at these levels.
# Target: 50-150 total trades over 4 years = 12-37/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # S1 = (2*P) - H
    # R1 = (2*P) - L
    # S2 = P - (H - L)
    # R2 = P + (H - L)
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    s1 = (2 * pivot) - weekly_high
    r1 = (2 * pivot) - weekly_low
    s2 = pivot - (weekly_high - weekly_low)
    r2 = pivot + (weekly_high - weekly_low)
    
    # Shift by 1 week to avoid look-ahead (use previous week's levels)
    pivot = np.roll(pivot, 1)
    s1 = np.roll(s1, 1)
    r1 = np.roll(r1, 1)
    s2 = np.roll(s2, 1)
    r2 = np.roll(r2, 1)
    pivot[0] = np.nan
    s1[0] = np.nan
    r1[0] = np.nan
    s2[0] = np.nan
    r2[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    
    # Calculate RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(rsi[i])):
            continue
        
        # Volume spike condition: current volume > 2x median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 2.0 * vol_median
        
        # Long conditions: price touches S1 or S2 with volume spike and RSI oversold
        if volume_spike and rsi[i] < 30:
            if (abs(close[i] - s1_aligned[i]) < 0.001 * close[i] or  # Within 0.1% of S1
                abs(close[i] - s2_aligned[i]) < 0.001 * close[i]):   # Within 0.1% of S2
                if position <= 0:
                    position = 1
                    signals[i] = base_size
        
        # Short conditions: price touches R1 or R2 with volume spike and RSI overbought
        elif volume_spike and rsi[i] > 70:
            if (abs(close[i] - r1_aligned[i]) < 0.001 * close[i] or  # Within 0.1% of R1
                abs(close[i] - r2_aligned[i]) < 0.001 * close[i]):   # Within 0.1% of R2
                if position >= 0:
                    position = -1
                    signals[i] = -base_size
        
        # Exit conditions: opposite touch or RSI returns to neutral zone
        elif position == 1:
            if (abs(close[i] - r1_aligned[i]) < 0.001 * close[i] or
                abs(close[i] - r2_aligned[i]) < 0.001 * close[i] or
                rsi[i] > 50):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            if (abs(close[i] - s1_aligned[i]) < 0.001 * close[i] or
                abs(close[i] - s2_aligned[i]) < 0.001 * close[i] or
                rsi[i] < 50):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Reversal_Volume_RSI"
timeframe = "6h"
leverage = 1.0