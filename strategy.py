#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Camarilla Pivot with Volume Filter
# Hypothesis: Daily Camarilla levels (R3/S3 and R4/S4) act as strong institutional barriers.
# Price breaking above R4 with volume indicates bullish continuation; breaking below S4 indicates bearish continuation.
# Price bouncing off R3/S3 with volume indicates mean reversion.
# Works in both bull and bear markets: In bull, breaks above R4 continue up; breaks below S3 get bought.
# In bear, breaks below S4 continue down; breaks above R3 get sold.
# Volume filter ensures only institutional participation triggers entries.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_daily_camarilla_pivot_volume_v1"
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
    
    # Get daily data for Camarilla calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    prev_daily_close[0] = prev_daily_close[1] if len(prev_daily_close) > 1 else 0
    
    # Calculate daily Camarilla pivot points
    daily_range = prev_daily_high - prev_daily_low
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r3 = daily_pivot + (daily_range * 1.1 / 2)
    daily_s3 = daily_pivot - (daily_range * 1.1 / 2)
    daily_r4 = daily_pivot + (daily_range * 1.1)
    daily_s4 = daily_pivot - (daily_range * 1.1)
    
    # Align to 12h timeframe (use previous day's levels)
    daily_r3_aligned = align_htf_to_ltf(prices, df_daily, daily_r3)
    daily_s3_aligned = align_htf_to_ltf(prices, df_daily, daily_s3)
    daily_r4_aligned = align_htf_to_ltf(prices, df_daily, daily_r4)
    daily_s4_aligned = align_htf_to_ltf(prices, df_daily, daily_s4)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(daily_r3_aligned[i]) or np.isnan(daily_s3_aligned[i]) or 
            np.isnan(daily_r4_aligned[i]) or np.isnan(daily_s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to S3 or volume drops
            if (close[i] <= daily_s3_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to R3 or volume drops
            if (close[i] >= daily_r3_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R4 with volume
            if ((high[i] > daily_r4_aligned[i] or high[i] > daily_r3_aligned[i]) and 
                (close[i] > daily_r4_aligned[i] or close[i] > daily_r3_aligned[i]) and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume
            elif ((low[i] < daily_s4_aligned[i] or low[i] < daily_s3_aligned[i]) and 
                  (close[i] < daily_s4_aligned[i] or close[i] < daily_s3_aligned[i]) and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals