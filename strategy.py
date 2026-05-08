#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter with 1w Williams Alligator trend confirmation.
# Long when Choppiness Index (14) > 61.8 (range) AND Williams Alligator Lips > Teeth (bullish) AND close > SMA(50).
# Short when Choppiness Index (14) > 61.8 (range) AND Williams Alligator Lips < Teeth (bearish) AND close < SMA(50).
# Exit when Choppiness Index (14) < 38.2 (trending) OR Alligator lines cross in opposite direction.
# This strategy captures mean reversion in ranging markets with trend confirmation to avoid false signals.
# The weekly Alligator filter ensures alignment with higher timeframe momentum.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Chop_Alligator_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price (H+L)/2
    median_price = (df_1w['high'] + df_1w['low']) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # Green line
    
    # Align Alligator lines to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Choppiness Index (14) on 1d data
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_14[0] = np.maximum(high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0]))  # Fix first value
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    
    # 1d SMA50 for trend filter
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(sma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]  # Bullish
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]  # Bearish
        
        # Choppiness regime
        chop_range = chop[i] > 61.8    # Ranging market
        chop_trend = chop[i] < 38.2    # Trending market
        
        if position == 0:
            # Long: ranging market + bullish Alligator + price above SMA50
            if chop_range and lips_above_teeth and close[i] > sma50[i]:
                signals[i] = 0.25
                position = 1
            # Short: ranging market + bearish Alligator + price below SMA50
            elif chop_range and lips_below_teeth and close[i] < sma50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trending market OR Alligator turns bearish
            if chop_trend or lips_below_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trending market OR Alligator turns bullish
            if chop_trend or lips_above_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals