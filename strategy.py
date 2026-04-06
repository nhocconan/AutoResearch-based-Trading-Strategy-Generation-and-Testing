#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Confirmation
Hypothesis: Weekly pivot levels act as strong support/resistance. Price breaking weekly R1/S1 with volume confirmation
captures momentum in both bull and bear markets. Fade at R2/S2 provides mean reversion in ranging markets.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14327_6h_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: require volume above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # Require 20% above average volume
    
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
    start = 20  # for volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price below S1 OR stoploss
            if close[i] <= s1_aligned[i] or close[i] <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above R1 OR stoploss
            if close[i] >= r1_aligned[i] or close[i] >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout of R1/S1 with volume OR fade at R2/S2
            long_breakout = (close[i] > r1_aligned[i]) and vol_filter[i]
            short_breakout = (close[i] < s1_aligned[i]) and vol_filter[i]
            long_fade = (close[i] < s2_aligned[i]) and vol_filter[i]
            short_fade = (close[i] > r2_aligned[i]) and vol_filter[i]
            
            if long_breakout or long_fade:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout or short_fade:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals