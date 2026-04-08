#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Pivot Reversal with Volume Confirmation v1
# Hypothesis: Price tends to revert to daily pivot points during range-bound markets (common in 2025).
# We fade at S3/R3 (strong support/resistance) and breakout at S4/R4.
# Volume confirms institutional participation at these key levels.
# Works in both bull/bear as it captures mean reversion and breakouts.
# Target: 12-37 trades/year (48-148 over 4 years).

name = "12h_daily_pivot_reversal_volume_v1"
timeframe = "12h"
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    daily_range = daily_high - daily_low
    s1 = 2 * pivot - daily_high
    s2 = pivot - daily_range
    s3 = s2 - daily_range
    s4 = s3 - daily_range
    r1 = 2 * pivot - daily_low
    r2 = pivot + daily_range
    r3 = r2 + daily_range
    r4 = r3 + daily_range
    
    # Align pivot levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume filter: current volume > 1.8x 24-period average (institutional interest)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(pivot_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(r3_12h[i]) or 
            np.isnan(s4_12h[i]) or np.isnan(r4_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or breaks below S3 (stop)
            if close[i] >= r3_12h[i] or close[i] <= s3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or breaks above R3 (stop)
            if close[i] <= s3_12h[i] or close[i] >= r3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Fade at S3/R3: price touches level and reverses
                # Long: price touches or goes below S3 but closes back above it
                if close[i] > s3_12h[i] and low[i] <= s3_12h[i]:
                    # Additional confirmation: price closing in upper half of daily range
                    daily_range = r3_12h[i] - s3_12h[i]
                    if daily_range > 0:
                        close_position = (close[i] - s3_12h[i]) / daily_range
                        if close_position > 0.5:  # Closing in upper half
                            position = 1
                            signals[i] = 0.25
                # Short: price touches or goes above R3 but closes back below it
                elif close[i] < r3_12h[i] and high[i] >= r3_12h[i]:
                    # Additional confirmation: price closing in lower half of daily range
                    daily_range = r3_12h[i] - s3_12h[i]
                    if daily_range > 0:
                        close_position = (close[i] - s3_12h[i]) / daily_range
                        if close_position < 0.5:  # Closing in lower half
                            position = -1
                            signals[i] = -0.25
                # Breakout continuation: price breaks S4/R4 with volume
                # Long breakout: price closes above S4
                elif close[i] > s4_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below R4
                elif close[i] < r4_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals