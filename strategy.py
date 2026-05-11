#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_Momentum
Hypothesis: Combines weekly pivot levels with Donchian breakout and momentum filter.
Long when: Price breaks above Donchian(20) high AND price > weekly pivot AND ADX > 20 (trending)
Short when: Price breaks below Donchian(20) low AND price < weekly pivot AND ADX > 20 (trending)
Exit when: Price crosses weekly pivot OR ADX < 15 (trend weakening)
Uses weekly pivot for institutional reference, Donchian for breakout, ADX for trend strength.
Designed to work in both bull (breakouts with momentum) and bear (trend continuation) markets.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
"""

name = "6h_WeeklyPivot_DonchianBreakout_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly pivot calculation ---
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # --- Donchian(20) breakout ---
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # --- ADX(14) for trend strength ---
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[0:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # --- Align weekly indicators to 6h ---
    pivot_6h = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_6h = align_htf_to_ltf(prices, df_weekly, r1)
    s1_6h = align_htf_to_ltf(prices, df_weekly, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(weekly pivot needs 1 bar, Donchian20, ADX14)
    start_idx = max(1, 20, 14*3)  # Wilder smoothing needs ~3*period for stability
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_6h[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Pivot filter
        above_pivot = close[i] > pivot_6h[i]
        below_pivot = close[i] < pivot_6h[i]
        
        # Trend strength filter
        trending = adx[i] > 20
        weak_trend = adx[i] < 15
        
        if position == 0:
            if breakout_up and above_pivot and trending:
                # Long: breakout above resistance with trend
                signals[i] = 0.25
                position = 1
            elif breakout_down and below_pivot and trending:
                # Short: breakout below support with trend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses pivot OR trend weakens
                if close[i] < pivot_6h[i] or weak_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses pivot OR trend weakens
                if close[i] > pivot_6h[i] or weak_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals