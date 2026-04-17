#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Regime
Hypothesis: Combine daily Camarilla R1/S1 breakouts with volume confirmation and 4h Chop filter to avoid whipsaws in sideways markets. 
The Chop index filters out low-trend environments, reducing false breakouts. Target low trade frequency (<50/year) for better generalization.
Works in bull/bear: breakouts capture momentum; Chop filter prevents losses in ranges.
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
    
    # === Daily data for Chop filter and pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Chop index: measures trend strength (high = range, low = trend)
    # Chop = 100 * log10(sum(ATR1) / (n * max(high-low))) / log10(n)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align length
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10(np.nansum(atr1) / ((hh - ll) * 14)) / np.log10(14)
    chop = np.where((hh - ll) > 0, chop_raw, 50)  # avoid division by zero
    
    # Chop > 61.8 = ranging (avoid), Chop < 38.2 = trending (favor breakouts)
    chop_filter = chop < 50  # Use median as threshold for simplicity
    
    # Chop aligned to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Daily Camarilla pivot levels (R1/S1) ===
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    r1 = pp + (range_hl * 1.1 / 12.0)
    s1 = pp - (range_hl * 1.1 / 12.0)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily volume average for confirmation ===
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers 20-day volume average and 14-day Chop
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = vol_1d_current > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Chop filter: only trade in trending markets (Chop < 50)
        chop_ok = chop_aligned[i] < 50
        
        # Entry: only when flat
        if position == 0:
            # Long: break above R1 + volume + chop filter
            if close[i] > r1_aligned[i] and vol_filter and chop_ok:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 + volume + chop filter
            elif close[i] < s1_aligned[i] and vol_filter and chop_ok:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit: reverse signal
        elif position == 1:
            if close[i] < s1_aligned[i]:  # break below S1 = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > r1_aligned[i]:  # break above R1 = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0