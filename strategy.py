#!/usr/bin/env python3
# 6h_WeeklyPivot_R4S4_Breakout_DailyTrend_Volume
# Hypothesis: Weekly pivot R4/S4 levels represent strong institutional breakout levels.
# Price breaking above R4 in an uptrend or below S4 in a downtrend signals strong momentum.
# Weekly timeframe provides institutional context, daily EMA34 filters trend direction,
# and volume confirmation (2x average) filters false breakouts.
# Works in bull markets (follows uptrends) and bear markets (follows downtrends)
# by only trading in direction of daily trend. Target: 15-30 trades/year.

name = "6h_WeeklyPivot_R4S4_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot levels (standard formula)
    # P = (H + L + C) / 3
    # R4 = R3 + (R2 - R1) = 3*P - 2*L
    # S4 = S3 - (S2 - S1) = 3*H - 2*P
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r4 = 3 * pivot_point - 2 * weekly_low
    weekly_s4 = 3 * weekly_high - 2 * pivot_point
    
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    
    # Volume confirmation (20-period MA on 6h = ~5 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA34 (34), weekly pivot (10), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(weekly_r4_aligned[i]) or 
            np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly R4 + volume
            if uptrend and close[i] > weekly_r4_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly S4 + volume
            elif downtrend and close[i] < weekly_s4_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R4
            if not uptrend or close[i] < weekly_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S4
            if not downtrend or close[i] > weekly_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals