#!/usr/bin/env python3
"""
6h Weekly Pivot Point Reversal with Volume Spike and Trend Filter
Hypothesis: Weekly pivot points (S3/R3 and S4/R4) act as strong support/resistance.
Price reversals at S3/R3 with volume confirmation capture mean reversion in ranging markets,
while breakouts of S4/R4 with volume and trend alignment capture momentum in trending markets.
Designed for 15-35 trades/year on 6h timeframe to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points and support/resistance levels."""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    _, _, _, r3_w, r4_w, _, _, s3_w, s4_w = calculate_pivot_points(
        df_w['high'].values, 
        df_w['low'].values, 
        df_w['close'].values
    )
    
    # Align weekly pivot levels to 6h timeframe
    r3_w_aligned = align_ltf_to_hlf(prices, df_w, r3_w)
    r4_w_aligned = align_ltf_to_hlf(prices, df_w, r4_w)
    s3_w_aligned = align_ltf_to_hlf(prices, df_w, s3_w)
    s4_w_aligned = align_ltf_to_hlf(prices, df_w, s4_w)
    
    # Volume spike: 2x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Trend filter: 50-period EMA on 6h
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r3_w_aligned[i]) or 
            np.isnan(r4_w_aligned[i]) or
            np.isnan(s3_w_aligned[i]) or
            np.isnan(s4_w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3 = r3_w_aligned[i]
        r4 = r4_w_aligned[i]
        s3 = s3_w_aligned[i]
        s4 = s4_w_aligned[i]
        
        if position == 0:
            # Long reversal at S3 with volume spike
            if price <= s3 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal at R3 with volume spike
            elif price >= r3 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            # Long breakout above R4 with volume and trend
            elif price > r4 and volume_spike[i] and price > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout below S4 with volume and trend
            elif price < s4 and volume_spike[i] and price < ema_50[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price reaches R3 (take profit) or breaks below S4 (stop)
            if price >= r3 or price < s4:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price reaches S3 (take profit) or breaks above R4 (stop)
            if price <= s3 or price > r4:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_S3R3_Reversal_S4R4_Breakout"
timeframe = "6h"
leverage = 1.0