#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Trend Filter v1
Hypothesis: Weekly pivot levels act as strong support/resistance. Price breaking above weekly R3 or below S3 with momentum (price > weekly EMA20) captures trending moves. Works in bull (breakouts above R3) and bear (breakdowns below S3). Weekly EMA20 filters counter-trend trades. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot and trend filter (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Weekly high, low, close for pivot calculation
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pp_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pp_w - low_w
    s1_w = 2 * pp_w - high_w
    r2_w = pp_w + (high_w - low_w)
    s2_w = pp_w - (high_w - low_w)
    r3_w = high_w + 2 * (pp_w - low_w)
    s3_w = low_w - 2 * (high_w - pp_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Weekly EMA20 for trend filter
    ema20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_prev = np.roll(ema20_w, 1)
    ema20_w_prev[0] = ema20_w[0]
    ema20_above = ema20_w > ema20_w_prev  # Rising EMA20 = bullish bias
    
    # Align weekly data to 6h timeframe
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    ema20_above_aligned = align_htf_to_ltf(prices, df_w, ema20_above)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For weekly EMA20 and pivots
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i]) or 
            np.isnan(ema20_above_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite pivot level break or stoploss
        if position == 1:  # long position
            # Exit: price breaks below weekly S3 OR stoploss
            if (close[i] <= s3_w_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above weekly R3 OR stoploss
            if (close[i] >= r3_w_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: pivot breakout + trend + volume
            bull_breakout = close[i] > r3_w_aligned[i]
            bear_breakout = close[i] < s3_w_aligned[i]
            
            bull_entry = bull_breakout and ema20_above_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and ema20_above_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
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