#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Trend Filter v1
Hypothesis: Weekly pivot levels act as strong support/resistance. Price breaking above R1 or below S1 with
weekly trend alignment captures momentum moves. Works in bull/bear via weekly EMA trend filter.
Targets 75-200 trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for pivot and trend (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly high, low, close for pivot calculation
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Weekly EMA50 for trend filter
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_prev = np.roll(weekly_ema50, 1)
    weekly_ema50_prev[0] = weekly_ema50[0]
    weekly_uptrend = weekly_ema50 > weekly_ema50_prev
    weekly_downtrend = weekly_ema50 < weekly_ema50_prev
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_downtrend)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 24-period EMA (4 days worth)
    vol_ema = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (weekly EMA50 needs 50 weeks)
    start = 350  # ~50 weeks of 6h data
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite pivot level break or stoploss
        if position == 1:  # long position
            # Exit: price breaks below S1 OR stoploss
            if (close[i] <= weekly_s1_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R1 OR stoploss
            if (close[i] >= weekly_r1_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: pivot break + trend + volume
            bull_breakout = close[i] > weekly_r1_aligned[i]
            bear_breakout = close[i] < weekly_s1_aligned[i]
            
            bull_entry = bull_breakout and weekly_uptrend_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and weekly_downtrend_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals