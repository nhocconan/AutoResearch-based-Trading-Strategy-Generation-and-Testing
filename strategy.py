#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_Trend_Filter
Hypothesis: Weekly pivot points (S1-S4, R1-R4) act as strong support/resistance levels. 
Breakouts above R4 or below S4 with volume confirmation and aligned daily trend (price vs daily EMA50) 
capture strong momentum moves. Designed for low trade frequency (15-30/year) to avoid fee drag,
works in both bull (buy breakouts) and bear (sell breakdowns) markets by following institutional levels.
"""

name = "6h_Weekly_Pivot_Breakout_Trend_Filter"
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
    
    # Calculate weekly pivot points from previous week
    # Use weekly high, low, close from 1 week ago
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC
    prev_week_high = df_1w['high'].iloc[-2]
    prev_week_low = df_1w['low'].iloc[-2]
    prev_week_close = df_1w['close'].iloc[-2]
    
    # Calculate pivot point and support/resistance levels
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot)
    r4 = r3 + (prev_week_high - prev_week_low)
    s4 = s3 - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe (constant for the week)
    # Create arrays of the same length as prices with pivot values
    r4_array = np.full(n, r4)
    s4_array = np.full(n, s4)
    r3_array = np.full(n, r3)
    s3_array = np.full(n, s3)
    
    # Daily trend filter: EMA 50
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA50
        if position == 0:
            # LONG: Price breaks above R4 with volume confirmation and above daily EMA50 (uptrend)
            if close[i] > r4_array[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 with volume confirmation and below daily EMA50 (downtrend)
            elif close[i] < s4_array[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below R3 or daily EMA50
            if close[i] < r3_array[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above S3 or daily EMA50
            if close[i] > s3_array[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals