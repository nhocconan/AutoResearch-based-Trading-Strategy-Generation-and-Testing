#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Filter_1W
Hypothesis: Williams Alligator (Jaws/Teeth/Lips) on 1d defines market regime; trade in direction of Alligator alignment on 1d with 1w trend filter. 
Long when Lips > Teeth > Jaws (bullish alignment) and price above Teeth; short when Lips < Teeth < Jaws (bearish alignment) and price below Teeth.
Use 1w EMA50 as trend filter to avoid counter-trend trades. Williams Alligator catches trends early; 1w filter improves robustness in bull/bear.
Target: 20-60 total trades over 4 years (5-15/year) with position size 0.25.
"""

name = "1d_WilliamsAlligator_Filter_1W"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1d data for Alligator (smoothed medians)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d: SMMA (Smoothed Moving Average) of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2.0
    median_price_vals = median_price.values
    
    # Jaws: SMMA(13, 8) - 13-period smoothed median, 8-period shift
    # Teeth: SMMA(8, 5) - 8-period smoothed median, 5-period shift
    # Lips: SMMA(5, 3) - 5-period smoothed median, 3-period shift
    def smma(arr, period, shift):
        """Smoothed Moving Average: EMA-like smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        # First value: simple average
        smma_vals = np.full_like(arr, np.nan)
        smma_vals[period-1] = np.mean(arr[:period])
        # Subsequent values: smoothed
        for i in range(period, len(arr)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
        # Apply shift
        smma_shifted = np.full_like(arr, np.nan)
        if shift < len(arr):
            smma_shifted[shift:] = smma_vals[:-shift]
        return smma_shifted
    
    jaws = smma(median_price_vals, 13, 8)
    teeth = smma(median_price_vals, 8, 5)
    lips = smma(median_price_vals, 5, 3)
    
    # Align Alligator lines to lower timeframe (1d->1d is identity, but keep for consistency)
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = multiplier * close_weekly[i] + (1 - multiplier) * ema50_weekly[i-1]
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_weekly_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaws_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaws_aligned[i])
        
        if position == 0:
            # Long: bullish Alligator alignment + price above Teeth + weekly uptrend
            if bullish_alignment and close[i] > teeth_aligned[i] and close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + price below Teeth + weekly downtrend
            elif bearish_alignment and close[i] < teeth_aligned[i] and close[i] < ema50_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks OR price crosses below Teeth
            if not bullish_alignment or close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks OR price crosses above Teeth
            if not bearish_alignment or close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals