#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot Reversal with Volume Filter
# Hypothesis: Camarilla pivot levels (L3/H3) act as strong intraday support/resistance.
# Price rejection at these levels with volume confirmation indicates institutional defense
# of key levels, leading to mean reversion. Works in both bull/bear markets:
# - In bull: price rejects at H3 and falls to pivot (profit taking)
# - In bear: price rejects at L3 and rises to panic (short covering)
# Target: 20-50 trades/year (80-200 over 4 years).

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
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels: based on previous day's range
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Camarilla formulas (based on previous day)
    # H4 = Close + 1.5 * (High - Low) * 1.1/2
    # H3 = Close + 1.0 * (High - Low) * 1.1/2
    # L3 = Close - 1.0 * (High - Low) * 1.1/2
    # L4 = Close - 1.5 * (High - Low) * 1.1/2
    # Pivot = (High + Low + Close) / 3
    
    # Calculate for previous day (shift by 1 to avoid look-ahead)
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else 0  # handle first value
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else 0
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else 0
    
    # Calculate Camarilla levels for previous day
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low) * 1.1 / 2.0
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low) * 1.1 / 2.0
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l3)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, camarilla_pivot)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price rises to pivot or volume drops
            if close[i] >= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price falls to pivot or volume drops
            if close[i] <= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price rejects at L3 (falls below then closes above) with volume
            if low[i] < l3_aligned[i] and close[i] > l3_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price rejects at H3 (rises above then closes below) with volume
            elif high[i] > h3_aligned[i] and close[i] < h3_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals