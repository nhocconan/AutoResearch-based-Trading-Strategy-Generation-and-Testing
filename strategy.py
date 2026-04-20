#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyPivot_TrendPullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly Pivot Points (previous week) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly aggregation: group by week (Sunday start)
    # We'll compute weekly pivot from the last completed week
    # For simplicity, we'll use the last 5 trading days as proxy for week
    # This is acceptable as we align and wait for completion
    
    # Calculate weekly high, low, close using 5-day rolling window
    # This approximates weekly OHLC (5 trading days per week)
    window = 5
    weekly_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
    weekly_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
    weekly_close = pd.Series(close_1d).rolling(window=window, min_periods=window).last().values
    
    # Shift by 1 to use previous week's data (lookback)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    
    # Weekly pivot point
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_range = prev_weekly_high - prev_weekly_low
    
    # Key weekly levels: S1 and R1 (using standard multiplier)
    # Weekly S1 = (2 * P) - High
    # Weekly R1 = (2 * P) - Low
    weekly_s1 = (2 * weekly_pivot) - prev_weekly_high
    weekly_r1 = (2 * weekly_pivot) - prev_weekly_low
    
    # Align weekly levels to 6h timeframe
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Trend and Pullback Logic ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 24-period EMA for trend (4 days on 6h chart)
    close_series = pd.Series(close)
    ema24 = close_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Pullback definition: price near weekly S1/R1 in trend direction
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        ema24_val = ema24[i]
        vol_ratio_val = vol_ratio[i]
        s1_val = weekly_s1_aligned[i]
        r1_val = weekly_r1_aligned[i]
        pivot_val = weekly_pivot_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema24_val) or np.isnan(vol_ratio_val) or 
            np.isnan(s1_val) or np.isnan(r1_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend (price > EMA24) and pullback to weekly S1
            if (close_val > ema24_val and  # Uptrend filter
                close_val >= s1_val * 0.995 and  # Near S1 (allow 0.5% slippage)
                close_val <= s1_val * 1.005 and
                vol_ratio_val > 1.3):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < EMA24) and pullback to weekly R1
            elif (close_val < ema24_val and  # Downtrend filter
                  close_val <= r1_val * 1.005 and  # Near R1
                  close_val >= r1_val * 0.995 and
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below weekly S1 or trend turns down
            if close_val < s1_val * 0.995 or close_val < ema24_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above weekly R1 or trend turns up
            if close_val > r1_val * 1.005 or close_val > ema24_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals