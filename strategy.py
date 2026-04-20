# 4h_MultiAsset_Structure_Regime
# Hypothesis: Trade breakouts of weekly structure (pivot points) with volume confirmation and regime filter (choppiness) on 4h timeframe.
# Weekly pivot points provide robust support/resistance levels that work across market regimes.
# Volume confirms breakout strength, choppiness filter avoids range-bound false signals.
# Works in bull/bear: pivot levels adapt to volatility, volume filters weak moves, regime filter adapts to market conditions.
# Target: 100-200 total trades over 4 years (25-50/year) with position size 0.25.

name = "4h_MultiAsset_Structure_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (more robust structure)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly Pivot: (H + L + C) / 3
    # R1: 2*P - L
    # S1: 2*P - H
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3
    r1_weekly = 2 * pivot_weekly - low_weekly
    s1_weekly = 2 * pivot_weekly - high_weekly
    
    # Align weekly pivot points to 4h
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for choppiness
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # ATR(14) for choppiness denominator
    atr_14 = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Sum of absolute daily changes for numerator
    abs_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_change = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        sum_abs_change[i] = np.sum(abs_change[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum(abs change)/ATR) / log10(14)
    chop = np.full_like(close_1d, np.nan)
    valid = (~np.isnan(sum_abs_change)) & (~np.isnan(atr_14)) & (atr_14 > 0)
    chop[valid] = 100 * np.log10(sum_abs_change[valid] / atr_14[valid]) / np.log10(14)
    
    # Align choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_weekly_aligned[i]) or np.isnan(r1_weekly_aligned[i]) or 
            np.isnan(s1_weekly_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when choppiness < 61.8 (trending market)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike AND trending regime
            if close[i] > r1_weekly_aligned[i] and volume_spike[i] and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike AND trending regime
            elif close[i] < s1_weekly_aligned[i] and volume_spike[i] and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly pivot OR choppiness > 61.8 (range)
            if close[i] < pivot_weekly_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly pivot OR choppiness > 61.8 (range)
            if close[i] > pivot_weekly_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals