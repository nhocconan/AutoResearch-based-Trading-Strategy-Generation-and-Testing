#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime_v1
Hypothesis: Use 1d Camarilla pivot levels (R1, S1) for breakout signals on 12h timeframe. 
Go long when price breaks above R1 with volume > 1.5x average and choppy market (CHOP > 61.8), 
short when price breaks below S1 with volume > 1.5x average and choppy market. 
Choppy market filter ensures mean-reversion behavior at pivot levels. 
Target: 12-30 trades/year by requiring confluence of breakout, volume, and regime.
Works in bull markets via breakouts and in bear via mean-reversion at pivot levels.
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) for each day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get weekly data for choppy market filter (CHOP)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Choppiness Index (CHOP) calculation
    chop_period = 14
    atr_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= chop_period:
        tr_1w = np.zeros_like(high_1w)
        for i in range(len(high_1w)):
            if i == 0:
                tr_1w[i] = high_1w[i] - low_1w[i]
            else:
                tr_1w[i] = max(
                    high_1w[i] - low_1w[i],
                    abs(high_1w[i] - close_1w[i-1]),
                    abs(low_1w[i] - close_1w[i-1])
                )
        
        # True Range moving sum
        tr_sum = np.full_like(close_1w, np.nan)
        for i in range(chop_period, len(tr_1w)):
            tr_sum[i] = np.sum(tr_1w[i-chop_period:i])
        
        # Highest high and lowest low over period
        hh_1w = np.full_like(high_1w, np.nan)
        ll_1w = np.full_like(low_1w, np.nan)
        for i in range(chop_period, len(high_1w)):
            hh_1w[i] = np.max(high_1w[i-chop_period:i])
            ll_1w[i] = np.min(low_1w[i-chop_period:i])
        
        # Chop calculation: 100 * log10(sum(tr1..tr14) / (HH - LL)) / log10(14)
        rr_1w = hh_1w - ll_1w
        chop_1w = np.full_like(close_1w, 50.0)  # default to neutral
        mask = (rr_1w > 0) & (~np.isnan(tr_sum))
        chop_1w[mask] = 100 * np.log10(tr_sum[mask] / rr_1w[mask]) / np.log10(chop_period)
    
    # Align Chop to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period) + 1  # Camarilla levels available from first bar
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(chop_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Choppy market filter: CHOP > 61.8 indicates ranging market (good for mean reversion at pivots)
        chop_filter = chop_12h[i] > 61.8
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and chop_filter:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_12h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_12h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (mean reversion) OR chop breaks down (trending market)
            if close[i] < s1_12h[i] or chop_12h[i] < 38.2:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (mean reversion) OR chop breaks down (trending market)
            if close[i] > r1_12h[i] or chop_12h[i] < 38.2:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0