#!/usr/bin/env python3
# 6h_1d1w_camarilla_trend_v1
# Hypothesis: 6h strategy using 1d Camarilla pivot levels filtered by 1w trend (EMA50 > EMA200).
# Long when: price > 1d R3, volume > 1.5x 20-period average, and 1w EMA50 > EMA200.
# Short when: price < 1d S3, volume > 1.5x 20-period average, and 1w EMA50 < EMA200.
# Exit: price returns to 1d pivot point (PP) or breaks opposite S4/R4 level.
# Uses 1d Camarilla for key support/resistance, 1w EMA for trend filter, 6h for execution.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d1w_camarilla_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = close_1d + range_1d * 1.1 / 4.0
    s3 = close_1d - range_1d * 1.1 / 4.0
    r4 = close_1d + range_1d * 1.1 / 2.0
    s4 = close_1d - range_1d * 1.1 / 2.0
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMAs for trend filter
    close_1w = df_1w['close'].values
    ema_fast = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slow = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_fast_aligned = align_htf_to_ltf(prices, df_1w, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1w, ema_slow)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(ema_fast_aligned[i]) or
            np.isnan(ema_slow_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: 1w EMA50 > EMA200 for long, < for short
        uptrend = ema_fast_aligned[i] > ema_slow_aligned[i]
        downtrend = ema_fast_aligned[i] < ema_slow_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to daily pivot or breaks below S4
            if close[i] <= pivot_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to daily pivot or breaks above R4
            if close[i] >= pivot_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume and trend confirmation
            bullish_breakout = (close[i] > r3_aligned[i]) and volume_confirmed and uptrend
            bearish_breakout = (close[i] < s3_aligned[i]) and volume_confirmed and downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals