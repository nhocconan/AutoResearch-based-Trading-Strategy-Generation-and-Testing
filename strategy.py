# -*- coding: utf-8 -*-
# -*- mode: python; -*-

#!/usr/bin/env python3
# 6H_WeeklyPivot_CounterTrend_With1DTrendFilter
# Hypothesis: 6-hour counter-trend strategy using weekly pivot levels with daily trend filter.
# Fades from weekly R2/S2 when price is in daily uptrend/downtrend, targeting mean reversion within weekly range.
# Works in bull markets (fades from weekly resistance in uptrend) and bear markets (fades from weekly support in downtrend).
# Targets 15-30 trades/year to minimize fee drag. Uses weekly structure with daily trend filter.

name = "6H_WeeklyPivot_CounterTrend_With1DTrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on previous week's OHLC)
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values
    
    # Calculate weekly pivot and support/resistance levels
    weekly_range = prev_weekly_high - prev_weekly_low
    pp_weekly = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    r2_weekly = pp_weekly + weekly_range * 0.25  # Weekly R2
    s2_weekly = pp_weekly - weekly_range * 0.25  # Weekly S2
    
    # Calculate daily EMA20 for trend filter
    ema_20_daily = pd.Series(prev_weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly levels and daily EMA to 6h timeframe
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2_weekly)
    pp_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    ema_20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have daily EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or 
            np.isnan(pp_weekly_aligned[i]) or np.isnan(ema_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects S2 support + daily uptrend (price > EMA20)
            if (close[i] <= s2_weekly_aligned[i] * 1.002 and  # Within 0.2% of S2
                close[i] > ema_20_daily_aligned[i]):          # Daily uptrend
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R2 resistance + daily downtrend (price < EMA20)
            elif (close[i] >= r2_weekly_aligned[i] * 0.998 and  # Within 0.2% of R2
                  close[i] < ema_20_daily_aligned[i]):         # Daily downtrend
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to weekly pivot (mean reversion target)
            at_pivot = abs(close[i] - pp_weekly_aligned[i]) < (r2_weekly_aligned[i] - pp_weekly_aligned[i]) * 0.3  # Within 30% of range to pivot
            
            if at_pivot:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals