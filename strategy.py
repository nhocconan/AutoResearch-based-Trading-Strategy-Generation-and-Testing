#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Pivot Reversal with Volume Filter
# Hypothesis: Daily pivot levels act as key support/resistance. Price rejecting at S1/R1 with
# volume confirmation indicates institutional defense of these levels, leading to mean reversion.
# Works in both bull/bear markets: reversals at S1 in bull (buy dips), reversals at R1 in bear (sell rallies).
# Target: 10-20 trades/year (40-80 over 4 years) to avoid overtrading.

name = "6h_daily_pivot_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily pivots: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_r1 = 2 * daily_pivot - daily_low
    daily_s1 = 2 * daily_pivot - daily_high
    
    # Align daily pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    
    # Volume filter: volume > 1.8x 30-period average (stricter to reduce trades)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches daily pivot or volume drops
            if close[i] >= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches daily pivot or volume drops
            if close[i] <= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price rejects S1 (closes above S1 after touching it) with volume
            if low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price rejects R1 (closes below R1 after touching it) with volume
            elif high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals