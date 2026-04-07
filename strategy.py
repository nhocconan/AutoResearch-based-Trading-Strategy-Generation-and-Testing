#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Volatility Breakout with Volume Filter
# Hypothesis: Weekly volatility compression (narrow range) followed by expansion
# with volume confirms institutional breakout. Works in bull/bear markets:
# - Bull: breakout above weekly high with volume = continuation
# - Bear: breakdown below weekly low with volume = continuation
# - Range: false breakouts filtered by volume requirement
# Target: 15-25 trades/year (60-100 over 4 years)

name = "1d_weekly_volatility_breakout_volume_v1"
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
    
    # Get weekly data for volatility calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly range (high-low) - use previous week to avoid look-ahead
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's completed data
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    # Handle first element
    if len(prev_weekly_high) > 1:
        prev_weekly_high[0] = prev_weekly_high[1]
        prev_weekly_low[0] = prev_weekly_low[1]
        prev_weekly_close[0] = prev_weekly_close[1]
    else:
        prev_weekly_high[0] = 0
        prev_weekly_low[0] = 0
        prev_weekly_close[0] = 0
    
    # Weekly range and position within range
    weekly_range = prev_weekly_high - prev_weekly_low
    # Avoid division by zero
    weekly_range = np.where(weekly_range == 0, 1e-10, weekly_range)
    weekly_position = (prev_weekly_close - prev_weekly_low) / weekly_range  # 0=at low, 1=at high
    
    # Align to daily timeframe
    weekly_position_aligned = align_htf_to_ltf(prices, df_weekly, weekly_position)
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, prev_weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, prev_weekly_low)
    
    # Volume filter: volume > 2.0x 20-day average for institutional participation
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_position_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to weekly midpoint or volume drops
            weekly_mid = (weekly_low_aligned[i] + weekly_high_aligned[i]) / 2.0
            if (close[i] <= weekly_mid or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price returns to weekly midpoint or volume drops
            weekly_mid = (weekly_low_aligned[i] + weekly_high_aligned[i]) / 2.0
            if (close[i] >= weekly_mid or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: weekly position shows strength (>0.7) and breaks above weekly high with volume
            if (weekly_position_aligned[i] > 0.7 and 
                high[i] > weekly_high_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: weekly position shows weakness (<0.3) and breaks below weekly low with volume
            elif (weekly_position_aligned[i] < 0.3 and 
                  low[i] < weekly_low_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals