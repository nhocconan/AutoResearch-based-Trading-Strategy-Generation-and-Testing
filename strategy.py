#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels and daily volatility regime.
# Weekly pivots provide key institutional levels; daily ATR regime filters for trending vs ranging.
# Long when price breaks above weekly R1 with daily ATR expansion (trending).
# Short when price breaks below weekly S1 with daily ATR expansion.
# In low volatility regime (ATR contraction), fade at weekly R2/S2 for mean reversion.
# Designed for low trade frequency (15-25/year) to work in both bull and bear markets.

name = "6h_WeeklyPivot_ATRRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard floor trader's method)
    pivot_w = np.zeros_like(close_w)
    r1_w = np.zeros_like(close_w)
    s1_w = np.zeros_like(close_w)
    r2_w = np.zeros_like(close_w)
    s2_w = np.zeros_like(close_w)
    
    for i in range(1, len(close_w)):
        # Previous week's high, low, close
        ph = high_w[i-1]
        pl = low_w[i-1]
        pc = close_w[i-1]
        
        # Pivot point and support/resistance levels
        pivot_w[i] = (ph + pl + pc) / 3.0
        r1_w[i] = 2 * pivot_w[i] - pl
        s1_w[i] = 2 * pivot_w[i] - ph
        r2_w[i] = pivot_w[i] + (ph - pl)
        s2_w[i] = pivot_w[i] - (ph - pl)
    
    # First week has no previous data
    pivot_w[0] = r1_w[0] = s1_w[0] = r2_w[0] = s2_w[0] = np.nan
    
    # Align weekly pivots to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Get daily data for ATR (volatility regime)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 14:
        return np.zeros(n)
    
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate True Range and ATR(14)
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with same length
    
    atr_14 = np.full_like(close_d, np.nan)
    for i in range(14, len(tr)):
        if np.isnan(tr[i-13:i+1]).any():
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Align ATR to 6h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_d, atr_14)
    
    # Calculate ATR ratio (current vs 20-period average) for regime detection
    atr_ma_20 = np.full_like(atr_14_aligned, np.nan)
    for i in range(20, len(atr_14_aligned)):
        if np.isnan(atr_14_aligned[i-20:i]).any():
            atr_ma_20[i] = np.nan
        else:
            atr_ma_20[i] = np.mean(atr_14_aligned[i-20:i])
    
    atr_ratio = atr_14_aligned / atr_ma_20
    # High volatility (trending) when ATR ratio > 1.2
    # Low volatility (ranging) when ATR ratio < 0.8
    vol_expanding = atr_ratio > 1.2
    vol_contracting = atr_ratio < 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for ATR calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(r2_w_aligned[i]) or
            np.isnan(s2_w_aligned[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Expanding volatility: trend following breakouts
            if vol_expanding[i]:
                if close[i] > r1_w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s1_w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # Contracting volatility: mean reversion at wider levels
            elif vol_contracting[i]:
                if close[i] > r2_w_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                elif close[i] < s2_w_aligned[i]:
                    signals[i] = 0.20
                    position = 1
        elif position == 1:
            # Long exit: volatility contraction and price at S1, or volatility expansion below pivot
            if vol_contracting[i] and close[i] < s1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif vol_expanding[i] and close[i] < pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: volatility contraction and price at R1, or volatility expansion above pivot
            if vol_contracting[i] and close[i] > r1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif vol_expanding[i] and close[i] > pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals