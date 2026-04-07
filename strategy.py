#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly 3-Point Reversal with Volume Confirmation
# Hypothesis: Weekly 3-point reversal patterns (break of prior week's high/low with
# reversal close) indicate institutional sentiment shifts. Combined with volume
# confirmation, this captures sustainable moves in both bull and bear markets.
# In bull markets: break above prior week's high + close > midpoint = bullish continuation.
# In bear markets: break below prior week's low + close < midpoint = bearish continuation.
# Uses only 2 conditions (price pattern + volume) to keep trades sparse (target: 15-35/year).
# Entry on close confirmation reduces whipsaw vs. intrabar breaks.

name = "6h_weekly_3pt_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Prior week's OHLC (shifted by 1 to avoid look-ahead)
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values
    
    # Handle first values
    if len(prev_weekly_high) > 1:
        prev_weekly_high[0] = prev_weekly_high[1]
        prev_weekly_low[0] = prev_weekly_low[1]
        prev_weekly_close[0] = prev_weekly_close[1]
    else:
        prev_weekly_high[0] = 0
        prev_weekly_low[0] = 0
        prev_weekly_close[0] = 0
    
    # Calculate midpoint of prior week's range
    weekly_midpoint = (prev_weekly_high + prev_weekly_low) / 2.0
    
    # Align weekly levels to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, prev_weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, prev_weekly_low)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_weekly, weekly_midpoint)
    
    # Volume filter: volume > 1.3x 20-period average (avoid churn)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_midpoint_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly midpoint or volume drops
            if (close[i] < weekly_midpoint_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above weekly midpoint or volume drops
            if (close[i] > weekly_midpoint_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above prior week's high AND closes above midpoint
            if ((high[i] > weekly_high_aligned[i]) and 
                (close[i] > weekly_midpoint_aligned[i]) and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below prior week's low AND closes below midpoint
            elif ((low[i] < weekly_low_aligned[i]) and 
                  (close[i] < weekly_midpoint_aligned[i]) and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals