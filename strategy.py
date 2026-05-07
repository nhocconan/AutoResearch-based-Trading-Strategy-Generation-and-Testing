#!/usr/bin/env python3
# 6h_Pivot_Fade_Trend
# Hypothesis: Combines pivot point mean reversion with trend filter to work in both bull and bear markets.
# On 6h chart, calculate daily pivot points and fade extreme levels (R3/S3) for mean reversion trades.
# Use 12h EMA50 as trend filter: only take longs when price > EMA50, shorts when price < EMA50.
# Add volume confirmation to avoid false signals. Designed for low trade frequency (~15-30/year).
# Works in ranging markets via pivot fade and trending markets via trend-aligned breakouts.
# Pivot points provide objective support/resistance levels that work across market regimes.
timeframe = "6h"
name = "6h_Pivot_Fade_Trend"
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
    
    # Calculate daily pivot points (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Trend filter: 12h EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price near S3 (strong support) + above EMA50 (uptrend) + volume spike
            if (close[i] <= s3_6h[i] * 1.005 and  # Allow 0.5% tolerance
                close[i] > ema_50_aligned[i] and
                volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price near R3 (strong resistance) + below EMA50 (downtrend) + volume spike
            elif (close[i] >= r3_6h[i] * 0.995 and  # Allow 0.5% tolerance
                  close[i] < ema_50_aligned[i] and
                  volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long
            # Exit if price reaches pivot (mean reversion target) or breaks below EMA50 (trend change)
            if (close[i] >= pivot_6h[i] * 0.995 or  # Near pivot
                close[i] < ema_50_aligned[i]):      # Below EMA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            # Exit if price reaches pivot (mean reversion target) or breaks above EMA50 (trend change)
            if (close[i] <= pivot_6h[i] * 1.005 or  # Near pivot
                close[i] > ema_50_aligned[i]):      # Above EMA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals