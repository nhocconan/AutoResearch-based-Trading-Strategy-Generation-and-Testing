#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume + Chop Regime
Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
Price approaching these levels with volume confirmation and in choppy market (range-bound) 
offers mean-reversion opportunities. Works in both bull and bear markets as price respects 
these levels during consolidation periods. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14436_12h_camarilla_pivot_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # PP = (High + Low + Close) / 3
    # R4 = Close + ((High - Low) * 1.1 / 2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    pp = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    r4 = close_1d + (range_hl * 1.1 / 2)
    r3 = close_1d + (range_hl * 1.1 / 4)
    r2 = close_1d + (range_hl * 1.1 / 6)
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    s2 = close_1d - (range_hl * 1.1 / 6)
    s3 = close_1d - (range_hl * 1.1 / 4)
    s4 = close_1d - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppy market filter using Choppy Index (EHLERS)
    def calculate_chop(high, low, close, window=14):
        """Calculate Choppy Index - values > 61.8 indicate ranging market"""
        atr_sum = pd.Series(high - low).rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        true_range = highest_high - lowest_low
        chop = 100 * np.log10(atr_sum / true_range) / np.log10(window)
        return chop.fillna(50).values  # neutral when undefined
    
    chop = calculate_chop(high, low, close, window=14)
    chop_filter = chop > 61.8  # choppy/ranging market
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # need enough data for indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check if we're in choppy market
        if not chop_filter[i]:
            # In trending market, reduce activity or go flat
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches S1 (stop loss) or R1 (take profit) or chop ends
            if (close[i] <= s1_aligned[i] or close[i] >= r1_aligned[i] or chop[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R1 (stop loss) or S1 (take profit) or chop ends
            if (close[i] >= r1_aligned[i] or close[i] <= s1_aligned[i] or chop[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean-reversion entries near pivot levels
            # Long when price touches/slightly penetrates S1 with rejection
            long_setup = (close[i] <= s1_aligned[i] * 1.002 and  # allow slight penetration
                         close[i] > s1_aligned[i] and  # but close above the level
                         vol_filter[i])
            
            # Short when price touches/slightly penetrates R1 with rejection
            short_setup = (close[i] >= r1_aligned[i] * 0.998 and  # allow slight penetration
                          close[i] < r1_aligned[i] and  # but close below the level
                          vol_filter[i])
            
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