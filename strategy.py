#!/usr/bin/env python3
"""
1d_WilliamsAlligator_ElderRay_TrendFollowing_v1
Concept: Williams Alligator identifies trend direction (Jaw/Teeth/Lips alignment),
Elder Ray measures bull/bear power for entry confirmation.
Long when: Alligator bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND price above Lips.
Short when: Alligator bearish (Lips < Teeth < Jaw) AND Bear Power < 0 AND price below Lips.
Exit when Alligator alignment breaks or power crosses zero.
Uses weekly trend filter to avoid counter-trend trades.
Conservative sizing (0.25) to manage drawdown in volatile markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WilliamsAlligator_ElderRay_TrendFollowing_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Daily Williams Alligator (SMMA-based) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    median_price = (high + low) / 2.0
    
    # Smoothed Moving Average (SMMA) - equivalent to RMA/Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Jaw (Blue) - 13-period SMMA of median price
    teeth = smma(median_price, 8)  # Teeth (Red) - 8-period SMMA of median price
    lips = smma(median_price, 5)   # Lips (Green) - 5-period SMMA of median price
    
    # === Elder Ray Index (using 13-period EMA as reference) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # === Weekly trend filter (EMA34) ===
    weekly_close = df_1w['close'].values
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Align daily indicators (they're already daily, but ensure alignment)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Identity alignment
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Get values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        weekly_ema34_val = weekly_ema34_aligned[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(bull_power_val) or np.isnan(bear_power_val) or 
            np.isnan(weekly_ema34_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment checks
        alligator_bullish = lips_val > teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long: Bullish Alligator alignment + Bull Power positive + price above Lips + above weekly EMA34
            if (alligator_bullish and bull_power_val > 0 and 
                close_val > lips_val and close_val > weekly_ema34_val):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + Bear Power negative + price below Lips + below weekly EMA34
            elif (alligator_bearish and bear_power_val < 0 and 
                  close_val < lips_val and close_val < weekly_ema34_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks OR Bull Power turns negative
            if not (alligator_bullish and bull_power_val > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks OR Bear Power turns positive
            if not (alligator_bearish and bear_power_val < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals