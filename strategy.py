#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime_v1
Hypothesis: Use 1d Camarilla pivot levels (R1/S1) for entry, with volume confirmation and 1d Choppiness Index regime filter. 
Go long when price breaks above 1d R1 with volume > 1.5x 20-period average and CHOP > 61.8 (range). 
Go short when price breaks below 1d S1 with volume > 1.5x 20-period average and CHOP > 61.8. 
Exit on opposite Camarilla level touch (S1 for long, R1 for short) or CHOP < 38.2 (trend). 
Position size: 0.25. Target: 20-40 trades/year by combining multiple filters to reduce noise and avoid overtrading.
Works in ranging markets via mean reversion at pivot levels and avoids trending markets via CHOP filter.
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
    
    # Get 1d data for Camarilla pivots and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 2
    # S1 = Pivot - (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 2.0
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d Choppiness Index (CHOP) - range: 0-100
    # CHOP = 100 * LOG10(SUM(ATR1) / (ATR(N) * N)) / LOG10(N)
    # We'll use a simplified version: CHOP = 100 * (ATR14 / (ATR14 * 14)) -> actually we need true range sum
    # Proper CHOP: 100 * log10(sum(tr1) / (atr14 * 14)) / log10(14)
    # Where tr1 = true range for 1 period
    
    # Calculate True Range for 1d
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    # Handle first period
    tr1[0] = high_1d[0] - low_1d[0]
    
    # ATR(14) - smoothed true range
    atr_period = 14
    atr_1d = np.full_like(close_1d, np.nan)
    
    if len(tr1) >= atr_period:
        # First ATR is simple average of first 14 TR
        atr_1d[atr_period-1] = np.mean(tr1[:atr_period])
        # Wilder smoothing for subsequent values
        for i in range(atr_period, len(tr1)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period - 1) + tr1[i]) / atr_period
    
    # Choppiness Index: 100 * log10(sum(tr1 over 14 periods) / (atr14 * 14)) / log10(14)
    chop_1d = np.full_like(close_1d, np.nan)
    log14 = np.log10(14)
    
    if len(tr1) >= atr_period:
        for i in range(atr_period, len(tr1)):
            # Sum of true range over last 14 periods
            tr_sum = np.sum(tr1[i-atr_period+1:i+1])
            # CHOP formula
            if atr_1d[i] > 0 and tr_sum > 0:
                chop_1d[i] = 100 * np.log10(tr_sum / (atr_1d[i] * atr_period)) / log14
            else:
                chop_1d[i] = 50.0  # neutral if calculation invalid
    
    # Align Choppiness Index to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Range regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        range_regime = chop_1d_aligned[i] > 61.8
        # Trend regime filter: CHOP < 38.2 indicates trending market (avoid)
        trend_regime = chop_1d_aligned[i] < 38.2
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and range_regime:
            # Long: price breaks above 1d R1 + volume + range regime
            if close[i] > r1_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d S1 + volume + range regime
            elif close[i] < s1_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches 1d S1 OR CHOP < 38.2 (trend emerging)
            if close[i] < s1_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches 1d R1 OR CHOP < 38.2 (trend emerging)
            if close[i] > r1_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0