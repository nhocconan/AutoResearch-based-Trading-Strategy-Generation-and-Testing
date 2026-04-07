#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Pivot Reversal with Volume Filter
# Hypothesis: Daily pivot points act as significant support/resistance levels where price often reverses.
# Price rejection at daily pivot with volume confirmation indicates institutional interest in these levels.
# Works in both bull and bear markets: in bull, price rejects above pivot and pulls back; in bear, price rejects below pivot and bounces.
# Target: 20-50 trades/year (80-200 over 4 years) by requiring clear rejection and volume confirmation.

name = "4h_daily_pivot_reversal_volume_v1"
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
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot: based on previous day's OHLC
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Use previous day's data to avoid look-ahead
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    # Handle first value
    if len(prev_high) > 1:
        prev_high[0] = prev_high[1]
        prev_low[0] = prev_low[1]
        prev_close[0] = prev_close[1]
    else:
        prev_high[0] = 0
        prev_low[0] = 0
        prev_close[0] = 0
    
    # Daily pivot point: (High + Low + Close) / 3
    daily_pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align pivot to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i]):
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
            # Long: price rejects below pivot (low < pivot and close > pivot) with volume
            if low[i] < pivot_aligned[i] and close[i] > pivot_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price rejects above pivot (high > pivot and close < pivot) with volume
            elif high[i] > pivot_aligned[i] and close[i] < pivot_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals