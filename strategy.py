#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with Volume Confirmation
# Hypothesis: At 6h timeframe, price reverts to mean after extreme deviations from
# daily Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout). Volume
# confirms institutional participation. Works in both bull and bear markets:
# - In bull: buy R3 bounces, avoid R4 breakdowns
# - In bear: sell S3 rallies, avoid S4 breakouts
# Target: 15-35 trades/year (60-140 over 4 years).

name = "6h_camarilla_pivot_reversal_volume_v1"
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
    
    # Calculate daily OHLC for pivot points
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Use previous day's data (shift by 1)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    
    # Handle first bar
    if len(prev_daily_high) > 1:
        prev_daily_high[0] = prev_daily_high[1]
        prev_daily_low[0] = prev_daily_low[1]
        prev_daily_close[0] = prev_daily_close[1]
    else:
        prev_daily_high[0] = 0
        prev_daily_low[0] = 0
        prev_daily_close[0] = 0
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # R3 = C + Range * 1.1/4
    # S3 = C - Range * 1.1/4
    # S4 = C - Range * 1.1/2
    pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    range_val = prev_daily_high - prev_daily_low
    
    r4 = pivot + range_val * 1.1 / 2.0
    r3 = pivot + range_val * 1.1 / 4.0
    s3 = pivot - range_val * 1.1 / 4.0
    s4 = pivot - range_val * 1.1 / 2.0
    
    # Align pivot levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion failure) or drops below S4 (strong breakdown)
            if low[i] < s3_aligned[i] or low[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion failure) or rises above R4 (strong breakout)
            if high[i] > r3_aligned[i] or high[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price touches/bounces off S3 with volume (mean reversion long)
            if ((low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) or
                (close[i] <= s3_aligned[i] and close[i] > s3_aligned[i] * 0.999)) and vol_filter[i]:
                # Additional filter: avoid if price broke below S4 (strong downtrend)
                if not (low[i] < s4_aligned[i]):
                    position = 1
                    signals[i] = 0.25
            # Short: price touches/rejects at R3 with volume (mean reversion short)
            elif ((high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) or
                  (close[i] >= r3_aligned[i] and close[i] < r3_aligned[i] * 1.001)) and vol_filter[i]:
                # Additional filter: avoid if price broke above R4 (strong uptrend)
                if not (high[i] > r4_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals