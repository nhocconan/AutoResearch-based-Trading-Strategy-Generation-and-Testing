#!/usr/bin/env python3
"""
6h Weekly Pivot Range Breakout with Volume Confirmation
Hypothesis: In ranging markets (common in 2025 BTC/ETH), price often respects weekly pivot levels.
Breakouts above weekly R1 or below S1 with volume confirmation indicate momentum continuation.
Weekly pivots provide structural support/resistance that works in both bull and bear markets.
Target: 15-25 trades/year to minimize fee drag while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly pivot data (higher timeframe than 6h)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point and support/resistance levels
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe (using previous week's values)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Volume confirmation: volume > 1.8x 24-period EMA (approx 6 days)
    vol_ema = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume
            if price > r1_aligned[i] and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume
            elif price < s1_aligned[i] and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to weekly pivot or S1
            if price < pp_aligned[i] or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly pivot or R1
            if price > pp_aligned[i] or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Range_Breakout"
timeframe = "6h"
leverage = 1.0