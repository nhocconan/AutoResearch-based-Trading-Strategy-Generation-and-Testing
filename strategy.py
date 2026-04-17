#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot R1/S1 Breakout with 1d Volume Spike and Chop Regime Filter.
Long when price breaks above R1 with volume spike and chop > 61.8 (ranging market).
Short when price breaks below S1 with volume spike and chop > 61.8.
Exit when price reverts to pivot point (PP) or chop < 38.2 (trending market).
Uses 1d for pivot calculation and ATR-based chop filter, 12h for execution.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels provide precise intraday support/resistance,
volume spike confirms participation, chop filter avoids false breakouts in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # PP = (High + Low + Close) / 3
    # R1 = PP + (High - Low) * 1.1 / 12
    # S1 = PP - (High - Low) * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pp_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = pp_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 12h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate ATR(20) for chop filter on 1d
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    high_close_1d[0] = high_low_1d[0]
    low_close_1d[0] = high_low_1d[0]
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Chopiness Index: CHOP = 100 * log10(sum(TR(14)) / (ATR(14) * 14)) / log10(14)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    chop_denominator = atr_14 * 14.0
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_ratio = sum_tr_14 / chop_denominator
    chop_ratio = np.where(chop_ratio <= 0, 1e-10, chop_ratio)
    chop_1d = 100.0 * np.log10(chop_ratio) / np.log10(14.0)
    
    # Align 1d ATR and Chop to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume spike: volume > 1.5 * volume MA(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(pp_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp = pp_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        chop = chop_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in choppy market (chop > 61.8 = ranging)
            if price > r1 and vol_spike and chop > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in choppy market (chop > 61.8 = ranging)
            elif price < s1 and vol_spike and chop > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point OR market starts trending (chop < 38.2)
            if price <= pp or chop < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR market starts trending (chop < 38.2)
            if price >= pp or chop < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0