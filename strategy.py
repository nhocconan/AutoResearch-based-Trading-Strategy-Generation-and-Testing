#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_VolumeSpike_RegimeFilter_V1
Hypothesis: 12h Camarilla pivot breakouts at R1/S1 with volume spike (>2.0x 20-period volume MA) and choppiness regime filter (CHOP < 50 = trending). 
12h timeframe reduces trade frequency to avoid fee drag. Camarilla levels from 1d HTF provide institutional support/resistance. 
Volume spike confirms institutional participation. Choppiness filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
Target: 12-37 trades/year (50-150 total over 4 years) with discrete position sizing (0.25) to minimize fee churn.
Uses 12h primary timeframe with 1d HTF for Camarilla calculation and 1w HTF for regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')  # for choppiness regime
    
    if len(df_1d) < 2 or len(df_1w) < 14:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12.0)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12.0)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1w Choppiness Index (CHOP) for regime filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum.reduce([tr1, tr2, tr3])])
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(atr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, atr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align CHOP to 12h timeframe (trending when CHOP < 50)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    trending_regime = chop_aligned < 50.0  # trending market
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation (>2x average)
        regime_ok = trending_regime[i]  # only trade in trending markets
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + trending regime
            if price > camarilla_r1_aligned[i] and vol_ok and regime_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + trending regime
            elif price < camarilla_s1_aligned[i] and vol_ok and regime_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (reversal) or volume spike fails
            if price < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (reversal) or volume spike fails
            if price > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeSpike_RegimeFilter_V1"
timeframe = "12h"
leverage = 1.0