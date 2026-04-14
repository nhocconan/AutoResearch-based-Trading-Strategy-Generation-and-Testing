#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator (3 SMAs) with daily pivot resistance/support levels
# Long when price > Alligator Jaw and price > Daily R1 pivot (bullish alignment)
# Short when price < Alligator Jaw and price < Daily S1 pivot (bearish alignment)
# Exit when price crosses Alligator Teeth (middle line)
# Uses daily pivot levels as dynamic support/resistance to avoid whipsaws in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Williams Alligator helps identify trend direction and strength in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 6h and daily data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 6h Williams Alligator: Jaw (13-period), Teeth (8-period), Lips (5-period) SMAs
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Alligator Jaw: 13-period SMMA of median price
    median_price_6h = (high_6h + low_6h) / 2
    jaw_raw = pd.Series(median_price_6h).rolling(window=13, min_periods=13).mean().values
    # Smoothed with 8-period shift (Williams Alligator specific)
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Alligator Teeth: 8-period SMMA of median price
    teeth_raw = pd.Series(median_price_6h).rolling(window=8, min_periods=8).mean().values
    # Smoothed with 5-period shift
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Alligator Lips: 5-period SMMA of median price
    lips_raw = pd.Series(median_price_6h).rolling(window=5, min_periods=5).mean().values
    # Smoothed with 3-period shift
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Calculate daily pivot points (standard formula)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    pivot = (high_daily + low_daily + close_daily) / 3
    r1 = 2 * pivot - low_daily
    s1 = 2 * pivot - high_daily
    r2 = pivot + (high_daily - low_daily)
    s2 = pivot - (high_daily - low_daily)
    
    # Align indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for 13-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price above Jaw AND above Daily R1 (bullish alignment)
            if (price > jaw_aligned[i] and 
                price > r1_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price below Jaw AND below Daily S1 (bearish alignment)
            elif (price < jaw_aligned[i] and 
                  price < s1_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth (middle line)
            if price < teeth_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Teeth (middle line)
            if price > teeth_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_DailyPivot"
timeframe = "6h"
leverage = 1.0