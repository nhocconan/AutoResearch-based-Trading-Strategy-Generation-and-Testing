#!/usr/bin/env python3
# 6h_Weekly_Pivot_Swing_Rejection_1dTrend
# Hypothesis: Fade at weekly pivot R4/S4 levels when price shows rejection (wick rejection) in the direction of the 1d EMA100 trend.
# Weekly pivots act as strong support/resistance on higher timeframe. Wick rejection indicates institutional defense of levels.
# Trend filter ensures trades align with weekly momentum, avoiding counter-trend trades in strong trends.
# Works in bull (buy R4 rejection in uptrend) and bear (sell S4 rejection in downtrend).
# Low frequency due to requirement of weekly pivot levels and clear price action rejection.

name = "6h_Weekly_Pivot_Swing_Rejection_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivots(high, low, close):
    """Calculate weekly pivot points: P, R1-S1, R2-S2, R3-S3, R4-S4"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivots
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Initialize pivot arrays
    r4 = np.full_like(wk_close, np.nan)
    s4 = np.full_like(wk_close, np.nan)
    
    # Calculate pivots for each weekly bar
    for i in range(len(wk_close)):
        if i == 0:
            # First bar: need previous week data, but we don't have it
            # Use current bar as placeholder (will be overwritten when real data available)
            pivot, r1, s1, r2, s2, r3, s3, r4_val, s4_val = calculate_weekly_pivots(
                wk_high[i], wk_low[i], wk_close[i]
            )
        else:
            pivot, r1, s1, r2, s2, r3, s3, r4_val, s4_val = calculate_weekly_pivots(
                wk_high[i-1], wk_low[i-1], wk_close[i-1]  # Use previous week's data
            )
        r4[i] = r4_val
        s4[i] = s4_val
    
    # Get daily data for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily trend: EMA100
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align weekly pivots and daily trend to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Wick rejection detection: long tail in opposite direction of move
    # Bullish rejection: long lower wick, close near high
    body_size = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    
    # Need open prices
    open_ = prices['open'].values
    
    bullish_rejection = (lower_wick > 2 * body_size) & (close > open_)  # Long lower wick, bullish close
    bearish_rejection = (upper_wick > 2 * body_size) & (close < open_)  # Long upper wick, bearish close
    
    # Align rejection signals
    bullish_rejection_aligned = align_htf_to_ltf(prices, prices, bullish_rejection.astype(float))
    bearish_rejection_aligned = align_htf_to_ltf(prices, prices, bearish_rejection.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema100_1d_aligned[i]) or
            np.isnan(bullish_rejection_aligned[i]) or
            np.isnan(bearish_rejection_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at S4 with bullish rejection + daily uptrend
            if close[i] <= s4_aligned[i] * 1.005 and bullish_rejection_aligned[i] and close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R4 with bearish rejection + daily downtrend
            elif close[i] >= r4_aligned[i] * 0.995 and bearish_rejection_aligned[i] and close[i] < ema100_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R2 or trend reversal
            # Calculate R2 for exit
            _, _, _, r2, _, _, _, _, _ = calculate_weekly_pivots(
                wk_high[0] if len(wk_high) > 0 else 0, 
                wk_low[0] if len(wk_low) > 0 else 0, 
                wk_close[0] if len(wk_close) > 0 else 0
            )
            # Simplified: exit at midpoint between S4 and R4 or trend change
            midpoint = (s4_aligned[i] + r4_aligned[i]) / 2
            if close[i] >= midpoint or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S2 or trend reversal
            midpoint = (s4_aligned[i] + r4_aligned[i]) / 2
            if close[i] <= midpoint or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals