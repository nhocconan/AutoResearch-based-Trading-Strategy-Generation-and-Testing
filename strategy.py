#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + Chop Regime
Hypothesis: 12h Camarilla pivot levels (L3/L4/S3/S4) with volume surge and chop regime filter
captures institutional breakouts and reversals. Works in bull (breakouts) and bear (reversals).
Target: 100-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot points (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    s1 = close_1d - (range_1d * 1.0833)
    s2 = close_1d - (range_1d * 1.1666)
    s3 = close_1d - (range_1d * 1.2500)
    s4 = close_1d - (range_1d * 1.5000)
    r1 = close_1d + (range_1d * 1.0833)
    r2 = close_1d + (range_1d * 1.1666)
    r3 = close_1d + (range_1d * 1.2500)
    r4 = close_1d + (range_1d * 1.5000)
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    vol_filter = volume > (2.0 * vol_ema)  # Require strong volume surge
    
    # Choppiness Index (14-period) for regime filter
    def calculate_chop(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        hhll = highest_high - lowest_low
        
        chop = np.zeros(len(close))
        for i in range(len(close)):
            if hhll[i] > 0 and atr[i] > 0:
                chop[i] = 100 * np.log10(sum(atr[max(0, i-period+1):i+1]) / hhll[i]) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop > 50.0  # Choppy/ranging market (mean reversion)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA and chop calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(vol_ema[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: mean reversion to pivot
        if position == 1:  # long position
            # Exit: price reaches pivot or S3/S4 levels
            if (close[i] <= pivot_aligned[i] or
                close[i] <= s3_aligned[i] or
                close[i] <= s4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches pivot or R3/R4 levels
            if (close[i] >= pivot_aligned[i] or
                close[i] >= r3_aligned[i] or
                close[i] >= r4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: S3/S4 bounce (long) or R3/R4 rejection (short) in choppy market
            long_setup = (close[i] <= s3_aligned[i] or close[i] <= s4_aligned[i]) and vol_filter[i] and chop_filter[i]
            short_setup = (close[i] >= r3_aligned[i] or close[i] >= r4_aligned[i]) and vol_filter[i] and chop_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals