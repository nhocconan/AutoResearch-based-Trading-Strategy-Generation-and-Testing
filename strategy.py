#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Alligator for trend direction and 1w Camarilla pivot levels for entry/exit.
# Long when price > Alligator Jaw (teeth) and breaks above R1 pivot with volume confirmation.
# Short when price < Alligator Jaw (teeth) and breaks below S1 pivot with volume confirmation.
# Uses 1w HTF for pivot calculation to reduce noise and increase trade quality.
# Designed for low trade frequency (12-25/year) to minimize fee drag while capturing trending moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d Indicators: Williams Alligator (13,8,5) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift forward by 8 bars
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift forward by 5 bars
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift forward by 3 bars
    
    # Use Jaw as the primary trend filter (slowest line)
    jaw_values = jaw.values
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_values)
    
    # === 1w Indicators: Camarilla Pivot Levels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for weekly timeframe
    pivot_1w = (high_1w[-1] + low_1w[-1] + close_1w[-1]) / 3.0
    range_1w = high_1w[-1] - low_1w[-1]
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    r1_1w = pivot_1w + (range_1w * 1.1 / 12)
    r2_1w = pivot_1w + (range_1w * 1.1 / 6)
    r3_1w = pivot_1w + (range_1w * 1.1 / 4)
    r4_1w = pivot_1w + (range_1w * 1.1 / 2)
    pp_1w = pivot_1w
    s1_1w = pivot_1w - (range_1w * 1.1 / 12)
    s2_1w = pivot_1w - (range_1w * 1.1 / 6)
    s3_1w = pivot_1w - (range_1w * 1.1 / 4)
    s4_1w = pivot_1w - (range_1w * 1.1 / 2)
    
    # Align the weekly pivot levels to 6h timeframe (constant values)
    # Since these are weekly levels, they remain constant throughout the week
    r1_aligned = np.full(n, r1_1w)
    r2_aligned = np.full(n, r2_1w)
    r3_aligned = np.full(n, r3_1w)
    r4_aligned = np.full(n, r4_1w)
    pp_aligned = np.full(n, pp_1w)
    s1_aligned = np.full(n, s1_1w)
    s2_aligned = np.full(n, s2_1w)
    s3_aligned = np.full(n, s3_1w)
    s4_aligned = np.full(n, s4_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(pp_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price above Alligator Jaw (uptrend)
        # 2. Price breaks above R1 pivot level
        # 3. Volume confirmation
        if (close[i] > jaw_aligned[i]) and (close[i] > r1_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price below Alligator Jaw (downtrend)
        # 2. Price breaks below S1 pivot level
        # 3. Volume confirmation
        elif (close[i] < jaw_aligned[i]) and (close[i] < s1_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Alligator_Jaw_Camarilla1w_R1S1_v1"
timeframe = "6h"
leverage = 1.0