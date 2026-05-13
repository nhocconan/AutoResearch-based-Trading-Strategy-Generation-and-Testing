#!/usr/bin/env python3
# 1d_Weekly_Pivot_R1_S1_Breakout_Trend_Volume
# Hypothesis: Weekly Pivot R1/S1 levels act as strong support/resistance on daily chart.
# Breakouts above R1 or below S1 with volume confirmation and weekly trend filter capture momentum.
# Uses 1d for execution and 1w EMA for trend direction. Target ~15-30 trades/year to avoid fee drag.
# Works in bull (breakouts with trend) and bear (breakdowns against trend filtered by 1w EMA).

name = "1d_Weekly_Pivot_R1_S1_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    pivot_1w = (h_1w + l_1w + c_1w) / 3.0
    weekly_r1 = 2 * pivot_1w - l_1w
    weekly_s1 = 2 * pivot_1w - h_1w
    
    # Align weekly Pivot levels to daily chart (wait for weekly close)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get weekly EMA trend filter
    ema_1w = pd.Series(c_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above weekly R1 with volume confirmation and weekly EMA uptrend
            if close[i] > weekly_r1_aligned[i] and volume_filter[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S1 with volume confirmation and weekly EMA downtrend
            elif close[i] < weekly_s1_aligned[i] and volume_filter[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly S1 or breaks below weekly EMA
            if close[i] < weekly_s1_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly R1 or breaks above weekly EMA
            if close[i] > weekly_r1_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals