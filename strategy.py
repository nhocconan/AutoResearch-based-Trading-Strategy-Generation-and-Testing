#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime_v1
Hypothesis: Use daily Camarilla pivot levels (R1, S1) for breakout signals on 4h timeframe.
Go long when price breaks above R1 with volume confirmation and favorable regime (Choppiness Index > 61.8 for mean-reversion or ADX < 20 for ranging).
Go short when price breaks below S1 with same filters.
Uses volume > 1.5x 20-period average for confirmation.
Designed to work in both bull and bear markets by adapting to regime: in ranging markets (high Chop), fade breakouts; in trending markets (low Chop), follow breakouts.
Target: 20-40 trades/year by combining multiple filters to avoid overtrading.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_width = 1.1 * (high_1d - low_1d) / 12
    r1_1d = close_1d + camarilla_width
    s1_1d = close_1d - camarilla_width
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get weekly data for regime filter (Choppiness Index)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Choppiness Index (14-period)
    chop_period = 14
    chop_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= chop_period + 1:
        atr_1w = np.full_like(close_1w, np.nan)
        for i in range(1, len(close_1w)):
            tr = max(
                high_1w[i] - low_1w[i],
                abs(high_1w[i] - close_1w[i-1]),
                abs(low_1w[i] - close_1w[i-1])
            )
            if i == 1:
                atr_1w[i] = tr
            else:
                atr_1w[i] = (atr_1w[i-1] * (chop_period - 1) + tr) / chop_period
        
        # Sum of true ranges over chop_period
        tr_sum = np.full_like(close_1w, np.nan)
        for i in range(chop_period, len(close_1w)):
            tr_sum[i] = np.sum(atr_1w[i-chop_period+1:i+1])
        
        # Choppiness Index formula
        for i in range(chop_period, len(close_1w)):
            if tr_sum[i] > 0:
                chop_1w[i] = 100 * np.log10(tr_sum[i] / (chop_period * atr_1w[i])) / np.log10(chop_period)
    
    # Align Choppiness Index to 4h timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, chop_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(chop_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: Choppiness Index > 61.8 = ranging (mean-revert), < 38.2 = trending
        chop_value = chop_1w_aligned[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        if position == 0 and vol_confirm:
            # In ranging market: fade breakouts (sell at R1, buy at S1)
            # In trending market: follow breakouts (buy at R1, sell at S1)
            if is_ranging:
                # Fade R1 (sell at resistance)
                if close[i] > r1_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                # Fade S1 (buy at support)
                elif close[i] < s1_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_trending:
                # Follow R1 breakout (buy)
                if close[i] > r1_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Follow S1 breakdown (sell)
                elif close[i] < s1_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches opposite S1 level or chop becomes extreme ranging
            if close[i] < s1_1d_aligned[i] or (chop_value > 70 and is_ranging):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches opposite R1 level or chop becomes extreme ranging
            if close[i] > r1_1d_aligned[i] or (chop_value > 70 and is_ranging):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0