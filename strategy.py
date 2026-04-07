#!/usr/bin/env python3
"""
4H Camarilla Pivot + Volume Spike + Choppiness Regime
Long when price touches S3 with volume spike in choppy market, short when touches R3
Exit at S4/R4 levels or when choppiness exits chop regime
Uses 1d Camarilla pivots for structure, volume for conviction, chop regime to avoid trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Choppiness Index (14) for regime filter ===
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    atr1[0] = 0
    atr2[0] = 0
    atr3[0] = 0
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    
    # True range sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    # Choppiness formula: 100 * log10(tr_sum / range_hl) / log10(14)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    
    # === 1D Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Camarilla levels
    range_prev = prev_high - prev_low
    camarilla_s3 = prev_close - (range_prev * 1.1 / 2)
    camarilla_s4 = prev_close - (range_prev * 1.1)
    camarilla_r3 = prev_close + (range_prev * 1.1 / 2)
    camarilla_r4 = prev_close + (range_prev * 1.1)
    
    # Align to 4h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(vol_ratio[i]) or np.isnan(chop[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S4 or chop exits chop regime (>61.8)
            if close[i] <= camarilla_s4_aligned[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 or chop exits chop regime (>61.8)
            if close[i] >= camarilla_r4_aligned[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Chop regime: chop < 61.8 (trending or chop, not strong trend)
            # Actually, we want chop > 61.8 for ranging market (mean reversion)
            if chop[i] <= 61.8:
                signals[i] = 0.0
                continue
            
            # Volume spike confirmation
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Camarilla S3/R3 touch with volume spike in choppy market
            # Long when price touches or goes below S3 with volume spike
            if close[i] <= camarilla_s3_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short when price touches or goes above R3 with volume spike
            elif close[i] >= camarilla_r3_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals