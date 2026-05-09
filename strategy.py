# Strategy Hypothesis:
# The 12h timeframe is well-suited for capturing multi-day trends while avoiding excessive noise.
# This strategy combines weekly pivot point breakouts with daily trend filtering and volume confirmation.
# Pivot points provide statistically significant support/resistance levels that often hold across market regimes.
# Weekly pivots offer stronger levels than daily pivots, suitable for 12h timeframe.
# Trend filter uses daily EMA to ensure trades align with higher timeframe momentum.
# Volume confirmation helps avoid false breakouts.
# Expected trade frequency: moderate (15-30 trades per year per symbol) to minimize fee drag.
# Works in both bull and bear markets because pivot levels adapt to price action and trend filter adapts to momentum.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyPivot_DailyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points (using previous week's OHLC)
    # Standard pivot point formula: P = (H + L + C) / 3
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Resistance levels
    R1 = 2 * pivot - prev_week_low
    R2 = pivot + (prev_week_high - prev_week_low)
    R3 = prev_week_high + 2 * (pivot - prev_week_low)
    # Support levels
    S1 = 2 * pivot - prev_week_high
    S2 = pivot - (prev_week_high - prev_week_low)
    S3 = prev_week_low - 2 * (prev_week_high - pivot)
    
    # Align weekly pivot levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1w, pivot)
    R1_12h = align_htf_to_ltf(prices, df_1w, R1)
    R2_12h = align_htf_to_ltf(prices, df_1w, R2)
    R3_12h = align_htf_to_ltf(prices, df_1w, R3)
    S1_12h = align_htf_to_ltf(prices, df_1w, S1)
    S2_12h = align_htf_to_ltf(prices, df_1w, S2)
    S3_12h = align_htf_to_ltf(prices, df_1w, S3)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: above 1.3x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_12h[i]) or np.isnan(R1_12h[i]) or np.isnan(R2_12h[i]) or np.isnan(R3_12h[i]) or
            np.isnan(S1_12h[i]) or np.isnan(S2_12h[i]) or np.isnan(S3_12h[i]) or
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.3 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 00-24 UTC (12h timeframe has fewer bars, so less restrictive)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = True  # No session filter for 12h to avoid missing opportunities
        
        if position == 0:
            # Long entry: price breaks above R1 with daily uptrend
            if (close[i] > R1_12h[i] and 
                close[i] > ema_50_12h[i] and  # Daily uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with daily downtrend
            elif (close[i] < S1_12h[i] and 
                  close[i] < ema_50_12h[i] and  # Daily downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below pivot (mean reversion to pivot)
            if close[i] < pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above pivot (mean reversion to pivot)
            if close[i] > pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals