#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily pivot direction and volume confirmation
# Uses 6h Donchian channel (20) for breakout signals, daily pivot points (from 1d) for trend direction,
# and 6h volume spike confirmation to filter false breakouts. Works in bull markets via breakout
# continuation and in bear markets via mean reversion at extreme pivot levels.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "6h_donchian20_daily_pivot_volume_v4"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 6h volume moving average and volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    vol_spike = vol_ratio > 1.5  # Volume 50% above average
    
    signals = np.zeros(n)
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # Trend filter from daily pivot: price above/below pivot indicates trend
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        
        # Mean reversion conditions at extreme pivot levels (S3/R3)
        at_s3 = close[i] <= s3_aligned[i] * 1.005  # Near S3 with small buffer
        at_r3 = close[i] >= r3_aligned[i] * 0.995  # Near R3 with small buffer
        
        # Volume confirmation
        vol_ok = vol_spike[i]
        
        # Long signals: bullish breakout with trend OR mean reversion at S3
        if (breakout_up and above_pivot and vol_ok) or (at_s3 and below_pivot and vol_ok):
            signals[i] = 0.25
        # Short signals: bearish breakout with trend OR mean reversion at R3
        elif (breakout_down and below_pivot and vol_ok) or (at_r3 and above_pivot and vol_ok):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals