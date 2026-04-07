# I am implementing a 1d strategy using weekly pivot points with volume confirmation.
# The hypothesis is that weekly pivot levels act as strong support/resistance levels.
# Price breaking above weekly R1 with volume indicates bullish continuation.
# Price breaking below weekly S1 with volume indicates bearish continuation.
# This should work in both bull and bear markets because:
# - In bull markets, breaks above R1 continue upward and breaks below S1 are bought as dips.
# - In bear markets, breaks below S1 continue downward and breaks above R1 are sold as rallies.
# The volume filter ensures only institutional participation triggers entries.
# Target: 7-25 trades per year (30-100 over 4 years).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_breakout_volume_v1"
timeframe = "1d"
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
    
    # Calculate weekly pivot points
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = (2 * weekly_pivot) - prev_weekly_low
    weekly_s1 = (2 * weekly_pivot) - prev_weekly_high
    weekly_r2 = weekly_pivot + (prev_weekly_high - prev_weekly_low)
    weekly_s2 = weekly_pivot - (prev_weekly_high - prev_weekly_low)
    
    # Align to 1d timeframe (use previous week's levels)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to weekly pivot or volume drops
            if (close[i] <= weekly_pivot_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to weekly pivot or volume drops
            if (close[i] >= weekly_pivot_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above weekly R1 with volume
            if ((high[i] > weekly_r1_aligned[i] or high[i] > weekly_r2_aligned[i]) and 
                (close[i] > weekly_r1_aligned[i] or close[i] > weekly_r2_aligned[i]) and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly S1 with volume
            elif ((low[i] < weekly_s1_aligned[i] or low[i] < weekly_s2_aligned[i]) and 
                  (close[i] < weekly_s1_aligned[i] or close[i] < weekly_s2_aligned[i]) and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals