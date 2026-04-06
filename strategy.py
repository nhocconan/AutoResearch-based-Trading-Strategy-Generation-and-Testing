#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with Volume Spike and Chop Filter
Hypothesis: Price tends to reverse at key intraday support/resistance levels (Camarilla pivots) derived from the prior day's range.
Institutional participation is confirmed by volume spikes. The Choppiness Index filters for trending regimes (avoid reversals in strong trends).
Works in bull markets (buy S3/S4, sell S5/S6) and bear markets (sell S5/S6, buy S3/S4) by fading extremes.
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_reversal_volume_chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # Camarilla formulas based on previous day's range
    range_1d = high_1d - low_1d
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = close_1d + (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    r3 = close_1d + (range_1d * 1.1 / 4)
    r4 = close_1d + (range_1d * 1.1 / 2)
    s1 = close_1d - (range_1d * 1.1 / 12)
    s2 = close_1d - (range_1d * 1.1 / 6)
    s3 = close_1d - (range_1d * 1.1 / 4)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align levels to 4h timeframe (use previous day's levels, shifted by 1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index (14) to detect trending vs ranging markets
    # Higher values (>61.8) indicate ranging, lower (<38.2) indicate trending
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # We want to fade extremes in ranging markets (chop > 50)
    chop_filter = chop > 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For chop and volume calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: mean reversion to pivot or stoploss
        if position == 1:  # long position
            # Exit: price reaches pivot (mean reversion) OR stoploss
            if (close[i] >= pivot_aligned[i] or
                close[i] <= entry_price - 2.5 * atr_sum[i]/14):  # approximate ATR
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches pivot (mean reversion) OR stoploss
            if (close[i] <= pivot_aligned[i] or
                close[i] >= entry_price + 2.5 * atr_sum[i]/14):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price at extreme levels + volume spike + chop filter
            # Long near S3/S4, Short near R3/R4
            long_setup = ((close[i] <= s3_aligned[i] * 1.001) or (close[i] <= s4_aligned[i] * 1.001)) and \
                         vol_spike[i] and chop_filter[i]
            short_setup = ((close[i] >= r3_aligned[i] * 0.999) or (close[i] >= r4_aligned[i] * 0.999)) and \
                          vol_spike[i] and chop_filter[i]
            
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