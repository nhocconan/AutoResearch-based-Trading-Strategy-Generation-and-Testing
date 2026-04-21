#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeATR_Regime_v1
Hypothesis: Breakout of Camarilla R1/S1 levels with volume confirmation and choppy regime filter.
Long when price breaks above R1 with volume spike and CHOP > 61.8 (rangy market favors mean reversion).
Short when price breaks below S1 with volume spike and CHOP > 61.8.
Exit when price reaches R2/S2 or reverses at R1/S1.
Works in both bull/bear by using 1d Camarilla levels and volume/regime filters to avoid false breakouts.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R2, S2
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    r2 = prev_close + 1.1 * rang / 6
    s2 = prev_close - 1.1 * rang / 6
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Choppiness Index (CHOP) on 1d
    if len(df_1d) >= 14:
        high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        atr_1d = np.abs(high_1d - low_1d)
        sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        range_14 = high_14 - low_14
        chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
        chop = np.where(range_14 == 0, 100, chop)  # avoid div by zero
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # default neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Regime filter: CHOP > 61.8 indicates rangy market (good for mean reversion)
        regime_ok = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long conditions: break above R1 with volume and rangy regime
            if (price > r1_aligned[i] and volume_ok and regime_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 with volume and rangy regime
            elif (price < s1_aligned[i] and volume_ok and regime_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reach R2 or reverse below R1
            if price >= r2_aligned[i] or price <= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reach S2 or reverse above S1
            if price <= s2_aligned[i] or price >= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeATR_Regime_v1"
timeframe = "4h"
leverage = 1.0