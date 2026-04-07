#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Monthly Pivot Breakout with Volume Filter
# Hypothesis: Monthly pivot levels are strong institutional support/resistance.
# Price breaking above monthly R1 with volume indicates bullish continuation.
# Price breaking below monthly S1 with volume indicates bearish continuation.
# Works in both bull and bear markets: In bull, breaks above R1 continue up; breaks below S1 get bought (mean reversion).
# In bear, breaks below S1 continue down; breaks above R1 get sold (mean reversion).
# Volume filter ensures only institutional participation triggers entries.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "6h_monthly_pivot_breakout_volume_v1"
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
    
    # Get monthly data for pivot calculation
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 2:
        return np.zeros(n)
    
    # Calculate monthly data (previous month's OHLC)
    monthly_high = df_monthly['high'].values
    monthly_low = df_monthly['low'].values
    monthly_close = df_monthly['close'].values
    
    # Shift by 1 to use previous month's data (avoid look-ahead)
    prev_monthly_high = np.roll(monthly_high, 1)
    prev_monthly_low = np.roll(monthly_low, 1)
    prev_monthly_close = np.roll(monthly_close, 1)
    prev_monthly_high[0] = prev_monthly_high[1] if len(prev_monthly_high) > 1 else 0
    prev_monthly_low[0] = prev_monthly_low[1] if len(prev_monthly_low) > 1 else 0
    prev_monthly_close[0] = prev_monthly_close[1] if len(prev_monthly_close) > 1 else 0
    
    # Calculate monthly pivot points
    monthly_pivot = (prev_monthly_high + prev_monthly_low + prev_monthly_close) / 3.0
    monthly_r1 = (2 * monthly_pivot) - prev_monthly_low
    monthly_s1 = (2 * monthly_pivot) - prev_monthly_high
    monthly_r2 = monthly_pivot + (prev_monthly_high - prev_monthly_low)
    monthly_s2 = monthly_pivot - (prev_monthly_high - prev_monthly_low)
    
    # Align to 6h timeframe (use previous month's levels)
    monthly_pivot_aligned = align_htf_to_ltf(prices, df_monthly, monthly_pivot)
    monthly_r1_aligned = align_htf_to_ltf(prices, df_monthly, monthly_r1)
    monthly_s1_aligned = align_htf_to_ltf(prices, df_monthly, monthly_s1)
    monthly_r2_aligned = align_htf_to_ltf(prices, df_monthly, monthly_r2)
    monthly_s2_aligned = align_htf_to_ltf(prices, df_monthly, monthly_s2)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(monthly_pivot_aligned[i]) or np.isnan(monthly_r1_aligned[i]) or 
            np.isnan(monthly_s1_aligned[i]) or np.isnan(monthly_r2_aligned[i]) or 
            np.isnan(monthly_s2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to monthly pivot or volume drops
            if (close[i] <= monthly_pivot_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to monthly pivot or volume drops
            if (close[i] >= monthly_pivot_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above monthly R1 with volume
            if ((high[i] > monthly_r1_aligned[i] or high[i] > monthly_r2_aligned[i]) and 
                (close[i] > monthly_r1_aligned[i] or close[i] > monthly_r2_aligned[i]) and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below monthly S1 with volume
            elif ((low[i] < monthly_s1_aligned[i] or low[i] < monthly_s2_aligned[i]) and 
                  (close[i] < monthly_s1_aligned[i] or close[i] < monthly_s2_aligned[i]) and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals