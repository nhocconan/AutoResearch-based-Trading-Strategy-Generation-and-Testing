#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_Breakout
Hypothesis: Price often reverses at weekly pivot levels (R1/S1) in ranging markets, but breaks through with momentum in trending markets. Uses weekly pivot points (calculated from prior week) as dynamic support/resistance. Entry: break of R1/S1 with volume confirmation (>1.5x 24-period average) and EMA50 filter (price > EMA for longs, < EMA for shorts) to avoid false breakouts. Exits on opposite pivot touch or EMA crossover. Designed for 6h timeframe to capture multi-day moves while avoiding excessive whipsaw. Weekly pivot provides robust levels that work in both bull and bear markets as it adapts to recent price action.
"""

name = "6h_Weekly_Pivot_Reversal_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for weekly pivot calculation (standard formula)
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Weekly Pivot Point (PP) and support/resistance levels
    pp = (ph + pl + pc) / 3.0
    r1 = 2 * pp - pl          # Resistance 1
    s1 = 2 * pp - ph          # Support 1
    r2 = pp + (ph - pl)       # Resistance 2
    s2 = pp - (ph - pl)       # Support 2
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # EMA50 on weekly close for trend filter
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike filter: current volume / 24-period average volume (24*6h = 6 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above R1 AND uptrend (price > EMA50) AND volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below S1 AND downtrend (price < EMA50) AND volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit conditions: touch S1 (contrarian exit) OR trend reversal (price < EMA50)
                if close[i] < s1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit conditions: touch R1 (contrarian exit) OR trend reversal (price > EMA50)
                if close[i] > r1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals