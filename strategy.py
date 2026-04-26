#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_Regime
Hypothesis: Trade 12h Camarilla R3/S3 breakouts with 1w EMA50 trend filter and ATR trailing stop.
12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
1w EMA50 ensures trading with dominant weekly trend.
ATR(14) trailing stop manages risk and captures trends.
Works in bull (breakouts with trend) and bear (mean reversion at extremes with trend filter).
"""

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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[0], tr1])  # align length
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of 1w EMA(50), ATR(14)
    start_idx = max(50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1w_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_1w_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above R3 AND 1w trend up
            long_signal = (close_val > r3_aligned[i]) and trend_1w_up
            
            # Short: price breaks below S3 AND 1w trend down
            short_signal = (close_val < s3_aligned[i]) and trend_1w_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # ATR trailing stop: exit if price drops 2.0 * ATR from highest since entry
            if close_val < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips down
            elif not trend_1w_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # ATR trailing stop: exit if price rises 2.0 * ATR from lowest since entry
            if close_val > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips up
            elif not trend_1w_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Regime"
timeframe = "12h"
leverage = 1.0