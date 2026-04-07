#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Daily Pivot Breakout with Volume Confirmation (tightened)
# Hypothesis: Daily pivot levels (R1/S1) act as strong support/resistance.
# Breakouts with volume confirmation indicate institutional participation.
# Uses 1h timeframe for entry timing, but signal direction from daily pivots.
# Works in bull/bear: breaks above R1 = long bias, breaks below S1 = short bias.
# Volume filter ensures breakouts have follow-through.
# Added 08-20 UTC session filter to reduce noise outside active trading hours.
# Target: 15-37 trades/year (60-150 over 4 years) to avoid fee drag.

name = "1h_daily_pivot_breakout_volume_v1"
timeframe = "1h"
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
    
    # Get daily data for pivot calculation
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
    
    # Calculate daily pivot points
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = (2 * daily_pivot) - prev_daily_low
    daily_s1 = (2 * daily_pivot) - prev_daily_high
    
    # Align to 1h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or 
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to pivot or volume drops
            if close[i] <= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to pivot or volume drops
            if close[i] >= pivot_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume (continuation in bull, mean reversion in bear)
            if high[i] > r1_aligned[i] and close[i] > r1_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.20
            # Short: price breaks below S1 with volume (continuation in bear, mean reversion in bull)
            elif low[i] < s1_aligned[i] and close[i] < s1_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.20
    
    return signals