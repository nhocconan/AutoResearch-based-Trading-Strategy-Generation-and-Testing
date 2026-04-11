#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with daily price action filter.
# Uses Alligator (Jaw/Teeth/Lips) from 12h for trend direction and entry signals.
# Enters long when Lips cross above Teeth in bullish alignment (Jaw < Teeth < Lips).
# Enters short when Lips cross below Teeth in bearish alignment (Jaw > Teeth > Lips).
# Filters trades by requiring price to be outside the Alligator's mouth (avoids chop).
# Designed for 12-37 trades/year on 6h with trend-following edge in both bull/bear markets.

name = "6h_12h_alligator_mouthfilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Alligator (SMMA-based) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_12h = (high_12h + low_12h) / 2  # Use median price
    
    # SMMA (Smoothed Moving Average) implementation
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(median_12h, 13)
    teeth = smma(median_12h, 8)
    lips = smma(median_12h, 5)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate Alligator mouth width (distance between highest and lowest of Jaw/Teeth/Lips)
    alligator_high = np.maximum(np.maximum(jaw_aligned, teeth_aligned), lips_aligned)
    alligator_low = np.minimum(np.minimum(jaw_aligned, teeth_aligned), lips_aligned)
    mouth_width = alligator_high - alligator_low
    
    # Average mouth width for filtering (20-period)
    mouth_avg = np.full_like(mouth_width, np.nan, dtype=float)
    for i in range(19, len(mouth_width)):
        mouth_avg[i] = np.mean(mouth_width[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(mouth_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Define Alligator alignment conditions
        bullish_alignment = (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i])
        
        # Define crossovers
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i] and lips_aligned[i-1] <= teeth_aligned[i-1]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i] and lips_aligned[i-1] >= teeth_aligned[i-1]
        
        # Price outside mouth filter: avoid trading when price is inside Alligator's mouth (chop)
        price_above_mouth = close[i] > alligator_high[i]
        price_below_mouth = close[i] < alligator_low[i]
        
        # Enter long: bullish alignment + lips crosses above teeth + price above mouth
        long_entry = bullish_alignment and lips_above_teeth and price_above_mouth
        
        # Enter short: bearish alignment + lips crosses below teeth + price below mouth
        short_entry = bearish_alignment and lips_below_teeth and price_below_mouth
        
        # Exit conditions: reversal of alignment or price re-enters mouth
        exit_long = (position == 1 and 
                    (not bullish_alignment or 
                     not price_above_mouth))
        exit_short = (position == -1 and 
                     (not bearish_alignment or 
                      not price_below_mouth))
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals