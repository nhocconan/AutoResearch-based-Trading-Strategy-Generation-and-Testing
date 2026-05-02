#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power with 1w trend filter
# Uses Williams Alligator (JAW/TEETH/LIPS) from 1w to define market structure and trend strength
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) from 1d to measure buying/selling pressure
# Long when: price > Alligator JAW AND Bull Power > 0 AND Bear Power weakening
# Short when: price < Alligator JAW AND Bear Power > 0 AND Bull Power weakening
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull markets (trend following with Alligator) and bear markets (mean reversion at extremes)
# BTC and ETH focused with SOL as validation

name = "6h_WilliamsAlligator_ElderRay_1w1d"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w Williams Alligator (SMMA = Smoothed Moving Average)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_1w = pd.Series(close_1w).ewm(alpha=1/13, adjust=False).mean().values
    jaw_1w = np.roll(jaw_1w, 8)
    jaw_1w[:8] = np.nan
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_1w = pd.Series(close_1w).ewm(alpha=1/8, adjust=False).mean().values
    teeth_1w = np.roll(teeth_1w, 5)
    teeth_1w[:5] = np.nan
    # Lips: 5-period SMMA, shifted 3 bars
    lips_1w = pd.Series(close_1w).ewm(alpha=1/5, adjust=False).mean().values
    lips_1w = np.roll(lips_1w, 3)
    lips_1w[:3] = np.nan
    
    # Align Alligator to 6h timeframe
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Calculate 1d Elder Ray Power (requires EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = EMA13 - Low
    bear_power_1d = ema13_1d - low_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Additional filter: 1w trend via Alligator alignment (all three lines ordered)
    # Bullish alignment: Lips > Teeth > Jaw
    # Bearish alignment: Jaw > Teeth > Lips
    bullish_align = (lips_1w_aligned > teeth_1w_aligned) & (teeth_1w_aligned > jaw_1w_aligned)
    bearish_align = (jaw_1w_aligned > teeth_1w_aligned) & (teeth_1w_aligned > lips_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price above JAW (bullish structure) AND Bull Power > 0 (buying pressure) 
            # AND Bullish Alligator alignment (trend confirmation)
            if (close[i] > jaw_1w_aligned[i] and 
                bull_power_1d_aligned[i] > 0 and 
                bullish_align[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price below JAW (bearish structure) AND Bear Power > 0 (selling pressure)
            # AND Bearish Alligator alignment (trend confirmation)
            elif (close[i] < jaw_1w_aligned[i] and 
                  bear_power_1d_aligned[i] > 0 and 
                  bearish_align[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price below JAW (structure break) OR Bear Power > Bull Power (momentum shift)
            if (close[i] < jaw_1w_aligned[i] or 
                bear_power_1d_aligned[i] > bull_power_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price above JAW (structure break) OR Bull Power > Bear Power (momentum shift)
            if (close[i] > jaw_1w_aligned[i] or 
                bull_power_1d_aligned[i] > bear_power_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals