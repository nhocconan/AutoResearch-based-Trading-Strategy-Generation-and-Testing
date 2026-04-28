#!/usr/bin/env python3
"""
6h_SeasonalTime_Filtered_Momentum
Hypothesis: Momentum strategy filtered by time-of-day and day-of-week to avoid low-liquidity periods and capture institutional flows.
Target: 15-35 trades/year per symbol with 0.25 position size. Works in both bull/bear by avoiding choppy periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 60-period ROC for momentum (60 * 6m = 240m = 4h)
    roc_period = 60
    roc = np.zeros_like(close)
    for i in range(roc_period, n):
        if close[i - roc_period] != 0:
            roc[i] = (close[i] - close[i - roc_period]) / close[i - roc_period]
    
    # Volume spike detector: 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute time filters
    hours = prices.index.hour
    day_of_week = prices.index.dayofweek  # Monday=0, Sunday=6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, roc_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        # Time filters: Avoid low liquidity periods
        hour = hours[i]
        dow = day_of_week[i]
        
        # Active periods: UTC 08:00-20:00 on weekdays, reduced on weekends
        is_weekday = dow < 5  # Monday-Friday
        is_active_hour = (8 <= hour <= 20)
        
        # Reduced weekend trading: only 12:00-18:00 UTC
        is_weekend_active = (dow >= 5) and (12 <= hour <= 18)
        
        time_filter = (is_weekday and is_active_hour) or is_weekend_active
        
        if not time_filter:
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Momentum condition with volume confirmation
        mom_long = roc[i] > 0.015  # 1.5% momentum
        mom_short = roc[i] < -0.015
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry conditions
        long_entry = time_filter and uptrend and mom_long and vol_confirm
        short_entry = time_filter and downtrend and mom_short and vol_confirm
        
        # Exit conditions: momentum fade or trend change
        long_exit = (roc[i] < 0.005) or (not uptrend)
        short_exit = (roc[i] > -0.005) or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_SeasonalTime_Filtered_Momentum"
timeframe = "6h"
leverage = 1.0