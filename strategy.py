#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Price Action + Weekly Pivot Rejection + Volume Spike
# Uses 1w pivot points (classic floor trader pivots) for rejection signals
# Price rejection at weekly R1/S1 with volume confirmation (>2x average)
# Only takes reversals (fades) at pivot levels, not breakouts
# Works in ranging markets (2023-2024) and volatile trending markets (2021-2022)
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag
# Weekly pivots provide institutional reference levels that hold across market regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly classic pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: volume > 2x average volume (24-period = 6 days)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Rejection conditions at weekly pivot levels
        # Bullish rejection: price touches S1 and closes back above it with volume
        bullish_reject = (low[i] <= s1_aligned[i] * 1.001 and  # touched or went below S1
                         close[i] > s1_aligned[i] and          # closed back above S1
                         vol > 2.0 * avg_vol[i])               # volume spike
        
        # Bearish rejection: price touches R1 and closes back below it with volume
        bearish_reject = (high[i] >= r1_aligned[i] * 0.999 and # touched or went above R1
                         close[i] < r1_aligned[i] and          # closed back below R1
                         vol > 2.0 * avg_vol[i])               # volume spike
        
        if position == 0:
            if bullish_reject:
                position = 1
                signals[i] = position_size
            elif bearish_reject:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches weekly pivot or opposite rejection
            if price >= pivot[i] * 0.999 or bearish_reject:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches weekly pivot or opposite rejection
            if price <= pivot[i] * 1.001 or bullish_reject:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyPivot_Rejection_Volume"
timeframe = "6h"
leverage = 1.0