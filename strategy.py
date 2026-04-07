#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend Filter
# Hypothesis: Daily Donchian(20) breakouts in direction of weekly trend capture
# momentum while avoiding whipsaws. Weekly trend filter ensures alignment with
# higher timeframe momentum, reducing false signals in choppy markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
# Works in bull markets (trend continuation) and bear markets (trend reversals).

name = "daily_donchian_breakout_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    daily_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    daily_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly EMA to daily timeframe
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema21)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(daily_high_20[i]) or np.isnan(daily_low_20[i]) or
            np.isnan(weekly_ema21_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below 20-day low or weekly trend turns bearish
            if close[i] < daily_low_20[i] or close[i] < weekly_ema21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above 20-day high or weekly trend turns bullish
            if close[i] > daily_high_20[i] or close[i] > weekly_ema21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above 20-day high with volume and bullish weekly trend
            if (high[i] > daily_high_20[i] and close[i] > daily_high_20[i] and
                vol_filter[i] and close[i] > weekly_ema21_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below 20-day low with volume and bearish weekly trend
            elif (low[i] < daily_low_20[i] and close[i] < daily_low_20[i] and
                  vol_filter[i] and close[i] < weekly_ema21_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals