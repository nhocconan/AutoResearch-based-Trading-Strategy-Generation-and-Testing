#!/usr/bin/env python3
# 12h_WeeklyPivot_R1S1_Breakout_1wTrend_VolumeConfirm
# Hypothesis: Weekly pivot points (R1/S1) provide key weekly support/resistance. Price breaking above R1 in a weekly uptrend or below S1 in a weekly downtrend indicates momentum. Volume confirmation filters false breakouts. Uses 12h timeframe to reduce trade frequency and avoid overtrading. Works in bull markets by riding uptrends and in bear markets by following downtrends.

name = "12h_WeeklyPivot_R1S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly pivot levels (standard formula)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * pivot_point - weekly_low
    weekly_s1 = 2 * pivot_point - weekly_high
    
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume confirmation (20-period MA on 12h = ~10 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation (stricter: >2.0x MA to reduce false signals)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly R1 + volume
            if uptrend and close[i] > weekly_r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly S1 + volume
            elif downtrend and close[i] < weekly_s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend or close[i] < weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend or close[i] > weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals