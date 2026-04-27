#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with weekly pivot filter and volume confirmation.
# Alligator identifies trends: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend.
# Uses weekly pivot (S1/S3/R1/R3) to filter trades: only long above S1, short below R1.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: 3 smoothed SMAs (5, 8, 13 periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = close_series.rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = close_series.rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Get weekly data for pivot filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (S1, S3, R1, R3)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    r3 = high_w + 2 * (pivot - low_w)
    s3 = low_w - 2 * (high_w - pivot)
    
    # Align weekly pivots to 6h timeframe (no extra delay needed for pivot points)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(pivot_s1_aligned[i]) or np.isnan(pivot_r1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Pivot filter: price above S1 for long, below R1 for short
        price_above_s1 = close[i] > pivot_s1_aligned[i]
        price_below_r1 = close[i] < pivot_r1_aligned[i]
        
        # Entry conditions with volume confirmation
        if alligator_long and price_above_s1 and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        elif alligator_short and price_below_r1 and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        else:
            # Exit conditions: Alligator reverses or pivot filter fails
            if position == 1:
                # Exit long if alligator turns bearish or price falls below S1
                if not alligator_long or not price_above_s1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if alligator turns bullish or price rises above R1
                if not alligator_short or not price_below_r1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1wPivot_VolumeFilter"
timeframe = "6h"
leverage = 1.0