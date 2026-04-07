#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Reversion with Volume Filter
# Hypothesis: Weekly pivot levels act as strong support/resistance.
# Price reverting from weekly R3/S3 with volume exhaustion indicates mean reversion opportunity.
# Works in both bull and bear:
# - In bull: pullbacks to weekly S3 find support; rallies to R3 face resistance
# - In bear: rallies to weekly R3 find resistance; drops to S3 find support
# Uses volume divergence (decreasing volume on approach) to confirm exhaustion.
# Target: 15-25 trades/year (60-100 over 4 years).

name = "6h_weekly_pivot_reversion_volume_v1"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly pivot points and S3/R3 levels
    # Pivot = (High + Low + Close) / 3
    # R3 = High + 2*(Pivot - Low)
    # S3 = Low - 2*(High - Pivot)
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align to 6h timeframe (use previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    
    # Volume filter: volume < 0.7x 20-period average (exhaustion)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_exhaustion = volume < (0.7 * vol_ma)
    
    # Price proximity: within 0.5% of S3/R3
    proximity_threshold = 0.005
    near_s3 = np.abs(close - s3_aligned) / s3_aligned < proximity_threshold
    near_r3 = np.abs(close - r3_aligned) / r3_aligned < proximity_threshold
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches pivot or volume returns
            if close[i] >= pivot_aligned[i] or not vol_exhaustion[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches pivot or volume returns
            if close[i] <= pivot_aligned[i] or not vol_exhaustion[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price near S3 with volume exhaustion (bullish reversal)
            if near_s3[i] and vol_exhaustion[i]:
                position = 1
                signals[i] = 0.25
            # Short: price near R3 with volume exhaustion (bearish reversal)
            elif near_r3[i] and vol_exhaustion[i]:
                position = -1
                signals[i] = -0.25
    
    return signals