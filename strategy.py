#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Pivot Breakout with Volume Filter and Chop Filter
# Hypothesis: Daily pivot levels act as strong support/resistance.
# Price breaking above R1 with volume and in trending market indicates continuation.
# Price breaking below S1 with volume and in trending market indicates continuation.
# Chop filter avoids whipsaws in ranging markets.
# Works in both bull and bear:
# - In bull: breaks above R1 continue up; breaks below S1 get bought (mean reversion)
# - In bear: breaks below S1 continue down; breaks above R1 get sold (mean reversion)
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_pivot_breakout_volume_chop_v1"
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
    
    # Align to 4h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Chop filter: Chop > 61.8 = ranging (avoid trades), Chop < 38.2 = trending (allow trades)
    atr_period = 14
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_close[1] if len(high_close) > 1 else 0
    low_close[0] = low_close[1] if len(low_close) > 1 else 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # True range for chop calculation
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((max_high - min_low) / atr_safe * np.sqrt(1/14)) / np.log10(14)
    chop_filter = chop < 61.8  # Allow trades when not extremely choppy
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to pivot or volume drops or chop too high
            if close[i] <= pivot_aligned[i] or not vol_filter[i] or not chop_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to pivot or volume drops or chop too high
            if close[i] >= pivot_aligned[i] or not vol_filter[i] or not chop_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume and in trending market
            if high[i] > r1_aligned[i] and close[i] > r1_aligned[i] and vol_filter[i] and chop_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with volume and in trending market
            elif low[i] < s1_aligned[i] and close[i] < s1_aligned[i] and vol_filter[i] and chop_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals