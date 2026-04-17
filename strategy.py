#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Camarilla R1/S1 breakout + volume confirmation + choppiness regime filter.
Long when price breaks above 12h Camarilla R1 with volume > 1.5x 20-period average and CHOP(14) < 38.2 (trending regime).
Short when price breaks below 12h Camarilla S1 with volume > 1.5x 20-period average and CHOP(14) < 38.2.
Exit when price reverts to the 12h Camarilla pivot point (PP).
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Choppiness filter ensures trades only occur in trending markets, reducing whipsaws in ranging conditions.
Works in bull markets (trend continuation) and bear markets (trend persistence after pullbacks).
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
    
    # Get 12h data for Camarilla pivot levels, volume, and choppiness
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous day)
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    pp = (high_12h + low_12h + close_12h) / 3.0
    r1 = pp + (high_12h - low_12h) * 1.1 / 12.0
    s1 = pp - (high_12h - low_12h) * 1.1 / 12.0
    
    # Calculate 12h volume 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Choppiness Index (CHOP) - 14 period
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)  # default to 50 (neutral) when undefined
    
    # Align all to 4h
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(volume_12h_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume_12h_aligned[i] > 1.5 * vol_ma_20_12h_aligned[i]
        
        # Choppiness regime filter: CHOP < 38.2 = trending regime (avoid ranging markets)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R1 with volume and trending regime
            if (close[i] > r1_aligned[i] and 
                volume_confirmed and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S1 with volume and trending regime
            elif (close[i] < s1_aligned[i] and 
                  volume_confirmed and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 12h Camarilla pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 12h Camarilla pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hCamarilla_R1S1_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0