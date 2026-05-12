#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3S3_Breakout_1wTrend
# Hypothesis: Use weekly trend filter (price above/below 200 EMA) with Camarilla R3/S3 breakout on 12h.
# Long when price breaks above R3 with weekly uptrend, short when breaks below S3 with weekly downtrend.
# Camarilla levels provide institutional support/resistance; weekly EMA filters trend direction.
# Designed for low frequency (15-35 trades/year) to avoid fee drag. Works in both bull and bear markets
# by aligning with weekly trend direction.

name = "12h_Camarilla_Pivot_R3S3_Breakout_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close.
    Returns R4, R3, R2, R1, PP, S1, S2, S3, S4.
    """
    typical = (high + low + close) / 3
    range_val = high - low
    R4 = close + range_val * 1.500
    R3 = close + range_val * 1.250
    R2 = close + range_val * 1.166
    R1 = close + range_val * 1.083
    PP = typical
    S1 = close - range_val * 1.083
    S2 = close - range_val * 1.166
    S3 = close - range_val * 1.250
    S4 = close - range_val * 1.500
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def calculate_ema(values, period):
    """Calculate EMA with proper handling of initial values."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    ema = np.full(len(values), np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    for i in range(period, len(values)):
        ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter (200 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = calculate_ema(close_1w, 200)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's data
    R4_1d, R3_1d, R2_1d, R1_1d, PP_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(
        high_1d[:-1], low_1d[:-1], close_1d[:-1]
    )
    
    # Shift to align with current day (today's levels based on yesterday's data)
    R4_1d = np.concatenate([[np.nan], R4_1d[:-1]])
    R3_1d = np.concatenate([[np.nan], R3_1d[:-1]])
    R2_1d = np.concatenate([[np.nan], R2_1d[:-1]])
    R1_1d = np.concatenate([[np.nan], R1_1d[:-1]])
    PP_1d = np.concatenate([[np.nan], PP_1d[:-1]])
    S1_1d = np.concatenate([[np.nan], S1_1d[:-1]])
    S2_1d = np.concatenate([[np.nan], S2_1d[:-1]])
    S3_1d = np.concatenate([[np.nan], S3_1d[:-1]])
    S4_1d = np.concatenate([[np.nan], S4_1d[:-1]])
    
    # Align daily Camarilla levels to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        weekly_downtrend = close[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R3 AND weekly uptrend
            if close[i] > R3_1d_aligned[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND weekly downtrend
            elif close[i] < S3_1d_aligned[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below R3 OR weekly trend turns down
            if close[i] < R3_1d_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S3 OR weekly trend turns up
            if close[i] > S3_1d_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals