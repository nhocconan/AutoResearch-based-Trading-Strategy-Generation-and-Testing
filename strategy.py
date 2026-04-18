#!/usr/bin/env python3
"""
1d Weekly Pivot Point Breakout with Volume Confirmation
Hypothesis: Price breaking above/below weekly pivot resistance/support levels 
(R1/S1) with volume confirmation (volume > 1.5x average) indicates strong momentum 
that persists across market regimes. Weekly pivots provide robust support/resistance 
levels that work in both bull and bear markets due to institutional order flow 
concentration at these levels. Target: 15-25 trades/year to minimize fee drag.
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
    
    # Get weekly high, low, close for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivot levels to daily timeframe (only use after weekly bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Volume confirmation: volume > 1.5x 20-day EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume EMA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Break above R1 with volume confirmation = long
            if price > r1_level and vol_conf:
                signals[i] = 0.25
                position = 1
            # Break below S1 with volume confirmation = short
            elif price < s1_level and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to pivot level
            if price < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to pivot level
            if price > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0