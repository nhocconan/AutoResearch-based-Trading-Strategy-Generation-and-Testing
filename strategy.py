#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Trend
12h strategy using Williams Alligator for trend direction and Williams %R for momentum confirmation.
- Long: Jaw < Teeth < Lips (bullish alignment) + %R crosses above -50 from below
- Short: Jaw > Teeth > Lips (bearish alignment) + %R crosses below -50 from above
- Exit: Opposite Alligator alignment
Williams Alligator: Jaw (13-period SMMA, 8 offset), Teeth (8-period SMMA, 5 offset), Lips (5-period SMMA, 3 offset)
Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in trending markets by catching trends early and avoiding whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def williams_r(high, low, close, period=14):
    """Williams %R indicator"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    wr = np.where(rr != 0, -100 * (highest_high - close) / rr, -50.0)
    return wr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Williams Alligator and Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator components
    # Jaw: 13-period SMMA of median price, 8 bars offset
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = smma(median_price_1d, 13)
    jaw_1d = np.roll(jaw_raw, 8)  # 8 bars forward offset
    jaw_1d[:8] = np.nan  # First 8 values invalid due to offset
    
    # Teeth: 8-period SMMA of median price, 5 bars offset
    teeth_raw = smma(median_price_1d, 8)
    teeth_1d = np.roll(teeth_raw, 5)  # 5 bars forward offset
    teeth_1d[:5] = np.nan  # First 5 values invalid due to offset
    
    # Lips: 5-period SMMA of median price, 3 bars offset
    lips_raw = smma(median_price_1d, 5)
    lips_1d = np.roll(lips_raw, 3)  # 3 bars forward offset
    lips_1d[:3] = np.nan  # First 3 values invalid due to offset
    
    # Williams %R (14-period)
    wr_1d = williams_r(high_1d, low_1d, close_1d, 14)
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for Williams %R and smoothed averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(wr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        bearish_alignment = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        # Williams %R momentum conditions
        wr_cross_up = wr_aligned[i] > -50 and wr_aligned[i-1] <= -50  # Cross above -50
        wr_cross_down = wr_aligned[i] < -50 and wr_aligned[i-1] >= -50  # Cross below -50
        
        if position == 0:
            # Long: bullish alignment + %R crosses above -50
            if bullish_alignment and wr_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + %R crosses below -50
            elif bearish_alignment and wr_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish alignment or %R crosses below -50
            if bearish_alignment or wr_cross_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish alignment or %R crosses above -50
            if bullish_alignment or wr_cross_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Trend"
timeframe = "12h"
leverage = 1.0