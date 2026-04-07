#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Pivot Breakout with Volume Filter and Trend Filter
# Hypothesis: Daily pivot levels act as strong support/resistance. Price breaking above R1 with volume and above EMA200 indicates bullish momentum, leading to continuation. Price breaking below S1 with volume and below EMA200 indicates bearish momentum, leading to continuation. The EMA200 filter ensures trades are taken in the direction of the long-term trend, reducing whipsaws in choppy markets. Works in both bull and bear markets: In bull, breaks above R1 with trend filter continue up; breaks below S1 get bought on pullbacks. In bear, breaks below S1 with trend filter continue down; breaks above R1 get sold on rallies. Volume filter ensures only institutional participation triggers entries.
# Target: 12-30 trades/year (48-120 over 4 years).

name = "12h_daily_pivot_breakout_volume_trend_v1"
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
    # Pivot = (High + Low + Close) / 3
    # R1 = (2 * Pivot) - Low
    # S1 = (2 * Pivot) - High
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = (2 * daily_pivot) - prev_daily_low
    daily_s1 = (2 * daily_pivot) - prev_daily_high
    
    # Align to 12h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    
    # EMA200 for trend filter (long-term trend)
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema200[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to S1 or trend turns bearish or volume drops
            if close[i] <= s1_aligned[i] or close[i] < ema200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to R1 or trend turns bullish or volume drops
            if close[i] >= r1_aligned[i] or close[i] > ema200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume and above EMA200 (bullish continuation)
            if high[i] > r1_aligned[i] and close[i] > r1_aligned[i] and vol_filter[i] and close[i] > ema200[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with volume and below EMA200 (bearish continuation)
            elif low[i] < s1_aligned[i] and close[i] < s1_aligned[i] and vol_filter[i] and close[i] < ema200[i]:
                position = -1
                signals[i] = -0.25
    
    return signals