#!/usr/bin/env python3
"""
160115: 6h_WeeklyPivot_DailyTrend_Flow
Hypothesis: Combines weekly pivot points (PP, R1, S1) with daily trend (EMA50) and volume flow confirmation on 6h timeframe.
Weekly pivots provide strong institutional support/resistance levels. Daily EMA50 filters for higher timeframe trend direction.
Volume flow (OBV slope) confirms institutional participation. Works in bull/bear by following weekly pivot structure with daily trend filter.
Targets 50-150 total trades over 4 years (12-37/year) with size 0.25.
"""

name = "6h_WeeklyPivot_DailyTrend_Flow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate weekly pivot points: PP, R1, S1
    # PP = (High + Low + Close) / 3
    # R1 = (2 * PP) - Low
    # S1 = (2 * PP) - High
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = (2 * pp) - low_1w
    s1 = (2 * pp) - high_1w

    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume confirmation: OBV slope > 0 (accumulation)
    # OBV = cumulative volume * sign(close - prev_close)
    close_series = pd.Series(close)
    price_change = close_series.diff()
    obv = (np.sign(price_change) * volume).cumsum()
    # OBV slope: 5-period linear regression slope
    def linreg_slope(arr, window):
        if len(arr) < window:
            return np.nan
        x = np.arange(window)
        y = arr[-window:]
        if np.all(np.isnan(y)):
            return np.nan
        slope = np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)[0] if np.sum(~np.isnan(y)) >= 2 else 0
        return slope
    
    obv_slope = np.full(len(obv), np.nan)
    for i in range(5, len(obv)):
        obv_slope[i] = linreg_slope(obv[:i+1], 5)
    obv_slope_pos = obv_slope > 0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(55, n):  # Start after EMA50 warmup
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(obv_slope_pos[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above PP + above S1 + daily uptrend + OBV accumulation
            if (close[i] > pp_aligned[i] and 
                close[i] > s1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                obv_slope_pos[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below PP + below R1 + daily downtrend + OBV distribution
            elif (close[i] < pp_aligned[i] and 
                  close[i] < r1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  not obv_slope_pos[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below PP (pivot breakdown) OR daily trend turns down
            if close[i] < pp_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above PP (pivot breakout) OR daily trend turns up
            if close[i] > pp_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals