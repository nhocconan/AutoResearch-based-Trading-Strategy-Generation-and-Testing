#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) + weekly pivot regime
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Weekly pivot provides regime: price above weekly pivot = bullish bias (long signals), below = bearish bias (short signals)
# Enter long when Bull Power > 0 and increasing (2-bar momentum) and price > weekly pivot
# Enter short when Bear Power < 0 and decreasing (2-bar momentum) and price < weekly pivot
# Uses discrete position sizing 0.25 to target ~12-25 trades/year and minimize fee drag
# Works in bull/bear markets: regime filter aligns with higher timeframe bias, Elder Ray captures momentum

name = "6h_1d_1w_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Calculate 1d Elder Ray momentum (2-bar change)
    bull_power_mom_1d = bull_power_1d - np.roll(bull_power_1d, 2)
    bear_power_mom_1d = bear_power_1d - np.roll(bear_power_1d, 2)
    # Handle first 2 bars
    bull_power_mom_1d[:2] = 0
    bear_power_mom_1d[:2] = 0
    
    # Load 1w data ONCE before loop for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*Pivot - L, S1 = 2*Pivot - H
    # R2 = Pivot + (H - L), S2 = Pivot - (H - L)
    # R3 = H + 2*(Pivot - L), S3 = L - 2*(H - Pivot)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    bull_power_mom_aligned = align_htf_to_ltf(prices, df_1d, bull_power_mom_1d)
    bear_power_mom_aligned = align_htf_to_ltf(prices, df_1d, bear_power_mom_1d)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(bull_power_mom_aligned[i]) or np.isnan(bear_power_mom_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if Bear Power becomes negative or price breaks below S1
            if bear_power_aligned[i] < 0 or close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Bull Power becomes positive or price breaks above R1
            if bull_power_aligned[i] > 0 or close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long signal: Bull Power positive AND increasing AND price above weekly pivot
            if (bull_power_aligned[i] > 0 and 
                bull_power_mom_aligned[i] > 0 and 
                close[i] > pivot_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Bear Power negative AND decreasing AND price below weekly pivot
            elif (bear_power_aligned[i] < 0 and 
                  bear_power_mom_aligned[i] < 0 and 
                  close[i] < pivot_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals