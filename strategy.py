#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Camarilla Pivot Reversion with Volume Filter
# Hypothesis: Price reverting from extreme Camarilla pivot levels (R4/S4) on daily timeframe
# with volume confirmation on 6h timeframe captures mean-reversion moves in both bull and bear markets.
# In bull markets: sell at R4 with volume, buy at S4 with volume.
# In bear markets: buy at S4 with volume, sell at R4 with volume.
# Daily pivots provide institutional reference levels, volume confirms participation.
# Target: 12-30 trades/year (50-120 over 4 years).

name = "6h_daily_camarilla_pivot_reversion_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate daily Camarilla pivot levels
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Pivot point and ranges
    pivot = (daily_high + daily_low + daily_close) / 3.0
    range_hl = daily_high - daily_low
    
    # Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Shift by 1 to use only completed daily bars (avoid look-ahead)
    r4 = np.roll(r4, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    
    # Handle first element
    if len(r4) > 1:
        r4[0] = r4[1]
        r3[0] = r3[1]
        s3[0] = s3[1]
        s4[0] = s4[1]
    else:
        r4[0] = r3[0] = s3[0] = s4[0] = 0
    
    # Align daily data to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above S3 (mean reversion complete) or volume filter fails
            if close[i] > s3_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses below R3 (mean reversion complete) or volume filter fails
            if close[i] < r3_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Short entry: price touches or crosses R4 with volume (sell the spike)
            if high[i] >= r4_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
            # Long entry: price touches or crosses S4 with volume (buy the dip)
            elif low[i] <= s4_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
    
    return signals