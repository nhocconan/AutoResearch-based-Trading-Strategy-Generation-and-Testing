#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h 12h Pivot Breakout with Volume Filter
# Hypothesis: 12-hour pivot levels act as key support/resistance. A break above R1 or below S1
# with volume confirmation indicates institutional participation and trend continuation.
# In bull markets, breaks above R1 continue upward; breaks below S1 often get bought (mean reversion).
# In bear markets, breaks below S1 continue downward; breaks above R1 often get sold (mean reversion).
# Volume filter ensures only significant participation triggers entries, reducing false breakouts.
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag.

name = "4h_12h_pivot_breakout_volume_v1"
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
    
    # Get 12h data for pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h data (previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift by 1 to use previous 12h bar's data (avoid look-ahead)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = prev_high_12h[1] if len(prev_high_12h) > 1 else 0
    prev_low_12h[0] = prev_low_12h[1] if len(prev_low_12h) > 1 else 0
    prev_close_12h[0] = prev_close_12h[1] if len(prev_close_12h) > 1 else 0
    
    # Calculate 12h pivot points
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    r1_12h = (2 * pivot_12h) - prev_low_12h
    s1_12h = (2 * pivot_12h) - prev_high_12h
    r2_12h = pivot_12h + (prev_high_12h - prev_low_12h)
    s2_12h = pivot_12h - (prev_high_12h - prev_low_12h)
    
    # Align to 4h timeframe (use previous 12h bar's levels)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(r2_12h_aligned[i]) or 
            np.isnan(s2_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to 12h pivot or volume drops
            if (close[i] <= pivot_12h_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to 12h pivot or volume drops
            if (close[i] >= pivot_12h_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above 12h R1 with volume
            if ((high[i] > r1_12h_aligned[i] or high[i] > r2_12h_aligned[i]) and 
                (close[i] > r1_12h_aligned[i] or close[i] > r2_12h_aligned[i]) and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 12h S1 with volume
            elif ((low[i] < s1_12h_aligned[i] or low[i] < s2_12h_aligned[i]) and 
                  (close[i] < s1_12h_aligned[i] or close[i] < s2_12h_aligned[i]) and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals