#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h 12h Camarilla Pivot + Volume Spike (Breakout at R4/S4)
# Hypothesis: Camarilla pivot levels (especially R4/S4) act as strong support/resistance.
# Breaking through R4/S4 with volume indicates institutional breakout.
# In bull markets: buy breakouts above R4 with volume confirmation.
# In bear markets: sell breakdowns below S4 with volume confirmation.
# Using 12h Camarilla levels calculated from prior day's OHLC.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "6h_12h_camarilla_pivot_breakout_v1"
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 12h bar
    # Using previous day's OHLC (shifted by 1)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift by 1 to use completed bar's OHLC for pivot calculation (avoid look-ahead)
    high_12h_prev = np.roll(high_12h, 1)
    low_12h_prev = np.roll(low_12h, 1)
    close_12h_prev = np.roll(close_12h, 1)
    
    # Handle first element
    if len(high_12h_prev) > 1:
        high_12h_prev[0] = high_12h_prev[1]
        low_12h_prev[0] = low_12h_prev[1]
        close_12h_prev[0] = close_12h_prev[1]
    else:
        high_12h_prev[0] = 0
        low_12h_prev[0] = 0
        close_12h_prev[0] = 0
    
    # Calculate pivot and Camarilla levels
    pivot = (high_12h_prev + low_12h_prev + close_12h_prev) / 3.0
    range_val = high_12h_prev - low_12h_prev
    
    # Camarilla levels
    r4 = close_12h_prev + range_val * 1.1 / 2
    r3 = close_12h_prev + range_val * 1.1 / 4
    s3 = close_12h_prev - range_val * 1.1 / 4
    s4 = close_12h_prev - range_val * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below R3 or volume filter fails
            if close[i] < r3_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above S3 or volume filter fails
            if close[i] > s3_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above R4 with volume
            if (high[i] > r4_aligned[i] and close[i] > r4_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below S4 with volume
            elif (low[i] < s4_aligned[i] and close[i] < s4_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals