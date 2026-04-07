#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Camarilla Pivot Reversal with Volume Filter
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong intraday support/resistance.
# Price rejecting these levels with volume exhaustion signals potential reversals.
# Works in bull/bear markets: sells at R3 in rallies, buys at S3 in declines.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "6h_daily_camarilla_pivot_reversal_volume_v1"
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
    
    # Get daily data for Camarilla pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    # P = (H+L+C)/3, Range = H-L
    # R3 = P + 1.1 * Range, S3 = P - 1.1 * Range
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    daily_r3 = daily_pivot + 1.1 * daily_range
    daily_s3 = daily_pivot - 1.1 * daily_range
    
    # Align daily Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_daily, daily_r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, daily_s3)
    
    # Volume filter: volume < 0.7x 20-period average (low volume on rejection)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume < (0.7 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches pivot or volume increases (breakout)
            if close[i] >= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches pivot or volume increases (breakdown)
            if close[i] <= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Short: price rejects R3 with low volume (bearish rejection)
            if close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
            # Long: price rejects S3 with low volume (bullish rejection)
            elif close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
    
    return signals