#!/usr/bin/env python3
"""
6h_WickReversal_With_Volume_and_Trend_Filter
Hypothesis: Trade long when price rejects lower wicks with volume spike in uptrend, short when rejects upper wicks in downtrend.
Uses 6h candle wicks (long lower shadow for long, long upper shadow for short) with volume > 1.5x 20-period average.
Trend filter: 1d EMA50 (uptrend if close > EMA50, downtrend if close < EMA50).
Designed for 6h to capture reversal points at support/resistance with confirmation.
Works in bull/bear: trend filter ensures trading with higher timeframe momentum.
Target: 50-120 total trades over 4 years (12-30/year) with position size 0.25.
"""

name = "6h_WickReversal_With_Volume_and_Trend_Filter"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Calculate wick conditions
    body_size = np.abs(close - open_) if 'open' in prices.columns else np.abs(close - np.roll(close, 1))
    # For first bar, use close-open approximation
    open_ = prices['open'].values if 'open' in prices.columns else np.roll(close, 1)
    open_[0] = close[0]  # approximate
    
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    
    # Long signal: long lower wick (> 2x body) with volume filter and uptrend
    long_condition = (lower_wick > 2 * body_size) & volume_filter & (close > ema50_1d_aligned)
    # Short signal: long upper wick (> 2x body) with volume filter and downtrend
    short_condition = (upper_wick > 2 * body_size) & volume_filter & (close < ema50_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(lower_wick[i]) or np.isnan(upper_wick[i]) or
            np.isnan(body_size[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            if long_condition[i]:
                signals[i] = 0.25
                position = 1
            elif short_condition[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below open (loss of bullish momentum) or trend turns down
            if close[i] < open_[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above open (loss of bearish momentum) or trend turns up
            if close[i] > open_[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals