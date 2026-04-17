#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Breakout at Camarilla R1 (long) or S1 (short) from daily pivot with volume confirmation and Choppiness regime filter.
Exit when price returns to central pivot (PP).
Designed to work in both bull and bear markets via regime filter that avoids trend-following in chop.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Pivot Points (using prior day's OHLC) ===
    # We'll calculate pivots from daily data and align to 12h
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for pivot calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot Point (PP) = (H + L + C)/3
    pp = (prev_high + prev_low + prev_close) / 3.0
    # R1 = 2*PP - L
    r1 = 2 * pp - prev_low
    # S1 = 2*PP - H
    s1 = 2 * pp - prev_high
    
    # Align daily pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Confirmation: 20-period volume average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Choppiness Index (14) for regime filter ===
    # CHOP = 100 * log10(sum(TR over n) / (n * (HHV - LLV))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hhll_diff = highest_high - lowest_low
    
    # Avoid division by zero
    chop = 100 * np.log10(atr_sum / (14 * hhll_diff + 1e-10)) / np.log10(14)
    # Replace invalid values (where hhll_diff == 0) with 50 (neutral)
    chop = np.where((hhll_diff == 0) | np.isnan(chop), 50.0, chop)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Regime filter: only trade when Choppiness < 61.8 (trending market)
        # Avoid chop where Choppiness > 61.8
        if chop[i] > 61.8:
            # In chop, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to central pivot (PP)
        elif position == 1:
            # Exit long: price crosses below PP
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above PP
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0