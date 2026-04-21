#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_ChopFilter_V1
Hypothesis: Camarilla R1/S1 breakouts with choppiness regime filter work on 12h for BTC/ETH in bull/bear markets. Uses 1d Camarilla levels, chop > 61.8 for ranging markets (mean reversion at S1/R1), chop < 38.2 for trending (breakouts). Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot point and R1/S1 levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12
    
    # Align daily levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Choppiness Index (14) on 12h close
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
    chop[hh == ll] = 100  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Chop > 61.8: ranging market - mean reversion at S1/R1
            if chop[i] > 61.8:
                # Long: price crosses above S1 (support) from below
                if i > 0 and close[i-1] <= s1_aligned[i-1] and price > s1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price crosses below R1 (resistance) from above
                elif i > 0 and close[i-1] >= r1_aligned[i-1] and price < r1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # Chop < 38.2: trending market - breakout continuation
            elif chop[i] < 38.2:
                # Long: price breaks above R1 with momentum
                if price > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 with momentum
                elif price < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reaches opposite level (R1) or chop shifts to strong trend
            if price >= r1_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches opposite level (S1) or chop shifts to strong trend
            if price <= s1_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0