#!/usr/bin/env python3
"""
1h_4h_Camarilla_Pivot_Breakout_1dTrend_Filter
Hypothesis: Camarilla pivot breakouts on 1h with 4h trend filter and 1d trend bias work in both bull and bear markets.
Breakout above R3 with 4h uptrend and 1d uptrend = long.
Breakdown below S3 with 4h downtrend and 1d downtrend = short.
Exit on opposite Camarilla level (S1 for longs, R1 for shorts).
Uses 4h and 1d for signal direction, 1h only for entry timing.
Target: 15-30 trades/year per symbol (60-120 over 4 years).
"""

name = "1h_4h_Camarilla_Pivot_Breakout_1dTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = ema_50_4h > 0  # placeholder, will align below
    downtrend_4h = ema_50_4h < 0  # placeholder
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = ema_50_1d > 0  # placeholder
    downtrend_1d = ema_50_1d < 0  # placeholder
    
    # Calculate Camarilla levels for each 1h bar using prior day's OHLC
    # We need to group by date and compute for each bar
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    # Arrays to store Camarilla levels for each 1h bar
    R3 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    
    # For each day, calculate levels from previous day's OHLC
    for i, date in enumerate(unique_dates):
        if i == 0:
            continue  # skip first day (no prior day)
        # Get prior day's data
        prior_date = unique_dates[i-1]
        prior_mask = (dates == prior_date)
        if not np.any(prior_mask):
            continue
        # Get OHLC of prior day
        prior_high = np.max(high[prior_mask])
        prior_low = np.min(low[prior_mask])
        prior_close = close[prior_mask][-1]  # last close of prior day
        
        # Calculate Camarilla levels
        range_val = prior_high - prior_low
        R3 = prior_close + range_val * 1.1 / 2
        S3 = prior_close - range_val * 1.1 / 2
        R1 = prior_close + range_val * 1.1 / 4
        S1 = prior_close - range_val * 1.1 / 4
        
        # Assign to today's bars
        today_mask = (dates == date)
        R3[today_mask] = prior_close + range_val * 1.1 / 2
        S3[today_mask] = prior_close - range_val * 1.1 / 2
        R1[today_mask] = prior_close + range_val * 1.1 / 4
        S1[today_mask] = prior_close - range_val * 1.1 / 4
    
    # Align 4h and 1d trends
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h)
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if Camarilla levels not available (first few days)
        if np.isnan(R3[i]) or np.isnan(S3[i]):
            signals[i] = 0.0
            continue
            
        r3 = R3[i]
        s3 = S3[i]
        r1 = R1[i]
        s1 = S1[i]
        uptrend_4h_val = uptrend_4h_aligned[i]
        downtrend_4h_val = downtrend_4h_aligned[i]
        uptrend_1d_val = uptrend_1d_aligned[i]
        downtrend_1d_val = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: break above R3, 4h uptrend, 1d uptrend
            if close[i] > r3 and uptrend_4h_val and uptrend_1d_val:
                signals[i] = 0.20
                position = 1
            # SHORT: break below S3, 4h downtrend, 1d downtrend
            elif close[i] < s3 and downtrend_4h_val and downtrend_1d_val:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 4h trend turns down
            if close[i] < s1 or not uptrend_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: touch R1 or 4h trend turns up
            if close[i] > r1 or not downtrend_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals