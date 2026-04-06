#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + Choppiness Regime
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Enter long at S3 support with volume spike in choppy market, short at R3 resistance.
Choppiness filter ensures we only mean-revert in ranging markets (CHOP > 61.8).
Works in bull by buying dips to support, in bear by selling rallies to resistance.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14422_12h_camarilla_pivot_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Resistance levels
    r1 = close_1d + (range_hl * 1.1 / 12)
    r2 = close_1d + (range_hl * 1.1 / 6)
    r3 = close_1d + (range_hl * 1.1 / 4)
    # Support levels
    s1 = close_1d - (range_hl * 1.1 / 12)
    s2 = close_1d - (range_hl * 1.1 / 6)
    s3 = close_1d - (range_hl * 1.1 / 4)
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index (14-period) - range detection
    def calculate_choppiness(high, low, close, period=14):
        atr_list = []
        for i in range(len(high)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_list.append(tr)
        
        atr_sum = pd.Series(atr_list).rolling(window=period, min_periods=period).sum()
        hh = pd.Series(high).rolling(window=period, min_periods=period).max()
        ll = pd.Series(low).rolling(window=period, min_periods=period).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        return chop.fillna(50).values  # fill NaN with 50 (neutral)
    
    chop = calculate_choppiness(high, low, close, 14)
    chop_filter = chop > 61.8  # Choppy/ranging market
    
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
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches S1 (take profit) OR stoploss OR chop breaks down
            if (close[i] <= s1_aligned[i] or 
                close[i] <= entry_price - 2.5 * atr[i] or
                chop[i] < 40):  # Trending market - exit mean reversion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R1 (take profit) OR stoploss OR chop breaks down
            if (close[i] >= r1_aligned[i] or 
                close[i] >= entry_price + 2.5 * atr[i] or
                chop[i] < 40):  # Trending market - exit mean reversion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price at S3/R3 + volume spike + choppy market
            long_setup = (close[i] <= s3_aligned[i] and vol_spike[i] and chop_filter[i])
            short_setup = (close[i] >= r3_aligned[i] and vol_spike[i] and chop_filter[i])
            
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