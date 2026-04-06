#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with 1d Trend Filter
Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) from 1d timeframe provide high-probability reversal zones.
Intraday (6b) price rejecting these levels with volume confirmation indicates institutional interest.
Trend filter ensures we only take reversals in direction of higher timeframe trend.
Works in ranging markets (reversions at R3/S3) and trending markets (breakouts at R4/S4).
Target: 60-120 trades over 4 years (~15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_reversal_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    # R3, S3 levels
    r3 = prev_close + range_1d * 1.1 / 2
    s3 = prev_close - range_1d * 1.1 / 2
    # R4, S4 levels (breakout zones)
    r4 = prev_close + range_1d * 1.1
    s4 = prev_close - range_1d * 1.1
    
    # Align Camarilla levels to 6b timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily trend filter: EMA50 vs EMA200
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_prev = np.roll(ema50_1d, 1)
    ema200_prev = np.roll(ema200_1d, 1)
    ema50_prev[0] = ema50_1d[0]
    ema200_prev[0] = ema200_1d[0]
    trend_up = ema50_1d > ema200_1d
    trend_down = ema50_1d < ema200_1d
    trend_up_6h = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_6h = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 24-period EMA (4 days worth)
    vol_ema = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For daily EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_ema[i]) or np.isnan(trend_up_6h[i]) or np.isnan(trend_down_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite signal or stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (target) or stoploss
            if (close[i] <= s3_6h[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (target) or stoploss
            if (close[i] >= r3_6h[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla reversals + trend + volume
            # Long setup: price rejects S3/S4 with volume, in uptrend
            long_reject_s3 = close[i] > s3_6h[i] and low[i] <= s3_6h[i]  # touched S3 but closed above
            long_break_s4 = close[i] > s4_6h[i]  # broke above S4 (breakout)
            long_setup = (long_reject_s3 or long_break_s4) and trend_up_6h[i] and volume[i] > vol_ema[i] * 1.5
            
            # Short setup: price rejects R3/R4 with volume, in downtrend
            short_reject_r3 = close[i] < r3_6h[i] and high[i] >= r3_6h[i]  # touched R3 but closed below
            short_break_r4 = close[i] < r4_6h[i]  # broke below R4 (breakdown)
            short_setup = (short_reject_r3 or short_break_r4) and trend_down_6h[i] and volume[i] > vol_ema[i] * 1.5
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals