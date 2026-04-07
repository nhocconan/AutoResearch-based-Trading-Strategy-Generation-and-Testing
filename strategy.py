#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot Reversal with Volume Filter
# Hypothesis: Price reacts at daily Camarilla pivot levels (S3/R3 for reversal) with volume confirmation.
# Works in bull markets (buy S3 reversals) and bear markets (sell R3 reversals).
# Target: 20-40 trades/year (80-160 over 4 years).

name = "4h_camarilla_pivot_reversal_volume_v1"
timeframe = "4h"
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
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Based on previous day's OHLC
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (daily_high + daily_low + daily_close) / 3.0
    range_hl = daily_high - daily_low
    
    # Camarilla levels (focus on S3 and R3 for reversals)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    pivot = np.roll(pivot, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    
    # Handle first element
    if len(pivot) > 1:
        pivot[0] = pivot[1]
        r3[0] = r3[1]
        s3[0] = s3[1]
    else:
        pivot[0] = 0
        r3[0] = 0
        s3[0] = 0
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: reversal at R3 or loss of volume
            if (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: reversal at S3 or loss of volume
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: reversal at S3 with volume
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: reversal at R3 with volume
            elif (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals