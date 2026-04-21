#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRSt
Hypothesis: 4h Camarilla pivot breakouts at R1/S1 with volume spike confirmation (>1.5x 20-period volume MA) and choppiness regime filter (CHOP > 61.8 for ranging markets). 
Camarilla levels from 1d HTF provide institutional support/resistance. Volume spikes confirm institutional participation. 
Choppiness filter avoids whipsaws in strong trends. Target 19-50 trades/year (75-200 total over 4 years).
Uses 4h primary timeframe with 1d HTF for Camarilla calculation and choppiness filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots and choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12.0)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d Choppiness Index (CHOP) ===
    atr_1d = []
    tr_1d = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])), np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # first TR is NaN
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d.append(np.nan)
        else:
            atr_1d.append(np.nanmean(tr_1d[i-13:i+1]))  # 14-period ATR
    atr_1d = np.array(atr_1d)
    
    # Sum of true range over 14 periods
    sum_tr_14 = np.convolve(tr_1d, np.ones(14), 'same')
    sum_tr_14[:13] = np.nan
    sum_tr_14[-13:] = np.nan
    
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (HH14 - LL14)) / log10(14)
    chop_1d = 100 * np.log10(sum_tr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_1d[(max_high_14 - min_low_14) == 0] = np.nan  # avoid division by zero
    
    # Align choppiness to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        chop_ok = chop_1d_aligned[i] > 61.8  # ranging market (mean reversion)
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + chop regime (range)
            if price > camarilla_r1_aligned[i] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + chop regime (range)
            elif price < camarilla_s1_aligned[i] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or volume spike fails or chop regime ends
            if price < camarilla_s1_aligned[i] or not vol_ok or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or volume spike fails or chop regime ends
            if price > camarilla_r1_aligned[i] or not vol_ok or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRSt"
timeframe = "4h"
leverage = 1.0