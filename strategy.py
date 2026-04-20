#!/usr/bin/env python3
# 6h_1d_Donchian_Weekly_Pivot_Breakout
# Hypothesis: Donchian(20) breakout on 6h with weekly pivot direction and volume confirmation.
# Weekly pivot provides directional bias: only take longs when above weekly pivot, shorts when below.
# Donchian breakout captures momentum, volume filter avoids false breakouts.
# Works in bull/bear: pivot adapts to regime, breakout captures trends in both directions.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Donchian_Weekly_Pivot_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # === Weekly Pivot Points ===
    # Using weekly high, low, close
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate pivot point and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Convert to numpy arrays
    pivot = np.asarray(pivot)
    r1 = np.asarray(r1)
    s1 = np.asarray(s1)
    r2 = np.asarray(r2)
    s2 = np.asarray(s2)
    r3 = np.asarray(r3)
    s3 = np.asarray(s3)
    
    # Align weekly pivot data to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    
    # === 6h: Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(pivot_val) or np.isnan(r1_val) or np.isnan(s1_val) or
            np.isnan(r2_val) or np.isnan(s2_val) or np.isnan(r3_val) or
            np.isnan(s3_val) or np.isnan(donchian_high_val) or
            np.isnan(donchian_low_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, above weekly pivot, volume confirmation
            if (high_val > donchian_high_val and  # Breakout above Donchian high
                close_val > pivot_val and         # Price above weekly pivot (bullish bias)
                vol_ratio_val > 1.5):             # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, below weekly pivot, volume confirmation
            elif (low_val < donchian_low_val and   # Breakout below Donchian low
                  close_val < pivot_val and        # Price below weekly pivot (bearish bias)
                  vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below Donchian low or below weekly pivot
            if low_val < donchian_low_val or close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above Donchian high or above weekly pivot
            if high_val > donchian_high_val or close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals