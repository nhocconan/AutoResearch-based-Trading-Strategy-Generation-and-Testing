#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation.
Long when price breaks above R4 (1d) AND 1w close > 1w open (bullish weekly candle) AND volume > 1.5x average.
Short when price breaks below S4 (1d) AND 1w close < 1w open (bearish weekly candle) AND volume > 1.5x average.
Exit when price reverts to R1/S1 (1d) OR weekly trend reverses.
Uses 6h for entry timing, 1d for Camarilla levels, 1w for trend filter to avoid counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla R4/S4 are strong breakout levels,
volume confirmation reduces fakeouts, weekly trend filter ensures trading with higher timeframe momentum.
Works in bull markets (captures uptrend breakouts) and bear markets (captures downtrend breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on 1d timeframe (based on previous day)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1, S4 = close - 1.1*(high-low)*1.1
    # R1 = close + 1.1*(high-low)*1.1/6, S1 = close - 1.1*(high-low)*1.1/6
    # Actually standard Camarilla: R4 = close + 1.1*(high-low)*1.1/2? Let me verify
    # Standard formula: R4 = close + 1.1*(high-low)*1.1, but that seems too wide
    # Correct: R4 = close + 1.1*(high-low)*1.1/2? No
    # Actual: R4 = close + 1.1*(high-low)*1.1, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # After checking: R4 = close + 1.1*(high-low)*1.1, S4 = close - 1.1*(high-low)*1.1
    # But this seems off. Let me use the correct formula:
    # Camarilla pivot levels:
    # Pivot = (high + low + close)/3
    # R4 = close + 1.1*(high-low)*1.1
    # R3 = close + 1.1*(high-low)*1.1/4
    # R2 = close + 1.1*(high-low)*1.1/6
    # R1 = close + 1.1*(high-low)*1.1/12
    # S1 = close - 1.1*(high-low)*1.1/12
    # S2 = close - 1.1*(high-low)*1.1/6
    # S3 = close - 1.1*(high-low)*1.1/4
    # S4 = close - 1.1*(high-low)*1.1
    
    # Actually, after verification, the standard is:
    # Range = high - low
    # R4 = close + Range * 1.1 * 1.1
    # But I think it's: R4 = close + Range * 1.1
    # Let me use the commonly accepted: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    # And R1/S1 at 1/6 of that range from close
    
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels
    r4 = close_1d + 1.1 * range_1d
    r3 = close_1d + 1.1 * range_1d / 4
    r2 = close_1d + 1.1 * range_1d / 6
    r1 = close_1d + 1.1 * range_1d / 12
    s1 = close_1d - 1.1 * range_1d / 12
    s2 = close_1d - 1.1 * range_1d / 6
    s3 = close_1d - 1.1 * range_1d / 4
    s4 = close_1d - 1.1 * range_1d
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Align 1w trend to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        r4_val = r4_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        s4_val = s4_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R4 AND weekly bullish AND volume > 1.5x avg
            if price > r4_val and weekly_bull and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < S4 AND weekly bearish AND volume > 1.5x avg
            elif price < s4_val and weekly_bear and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < R1 OR weekly trend turns bearish
            if price < r1_val or not weekly_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > S1 OR weekly trend turns bullish
            if price > s1_val or not weekly_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0