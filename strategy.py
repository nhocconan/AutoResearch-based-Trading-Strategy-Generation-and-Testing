#!/usr/bin/env python3
"""
12h Pivot Breakout with 1w Trend Filter
Long: Price breaks above R3 (weekly pivot) + weekly close > weekly open (bullish weekly candle)
Short: Price breaks below S3 (weekly pivot) + weekly close < weekly open (bearish weekly candle)
Exit: Price returns to weekly pivot (PP)
Uses weekly pivot levels from 1w timeframe for structure, with same-week confirmation.
Designed to capture strong breakouts in trending markets while avoiding false breakouts in chop.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # PP = (H + L + C) / 3
    # R3 = H + 2*(PP - L)  [or: PP + 2*(H - L)]
    # S3 = L - 2*(H - PP)  [or: PP - 2*(H - L)]
    # Using alternative formulas to avoid dependency on prior close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r3 = weekly_high + 2.0 * (pp - weekly_low)  # R3 = H + 2*(PP - L)
    s3 = weekly_low - 2.0 * (weekly_high - pp)  # S3 = L - 2*(H - PP)
    
    # Weekly bullish/bearish candle filter
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly levels to 12h timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 1  # Need at least one aligned value
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: Price breaks above R3 + weekly bullish candle
            if price > r3_val and weekly_bull:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + weekly bearish candle
            elif price < s3_val and weekly_bear:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below weekly pivot (PP)
            if price <= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above weekly pivot (PP)
            if price >= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_Breakout_1wTrend"
timeframe = "12h"
leverage = 1.0