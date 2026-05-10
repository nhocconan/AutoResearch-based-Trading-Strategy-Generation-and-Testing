#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels from weekly timeframe provide strong institutional support/resistance.
# In trending markets (1w EMA50), price breaking above weekly R3 in uptrend or below weekly S3 in downtrend continues with momentum.
# Volume confirmation filters false breakouts. Works in bull markets (follows uptrends) and bear markets (follows downtrends)
# by only trading in direction of weekly trend. Uses 12h timeframe to reduce trade frequency and fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Get weekly data for trend filter and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly Camarilla pivot levels
    # Standard formula for timeframe: H, L, C from previous week
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point calculation
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    range_hl = weekly_high - weekly_low
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    weekly_r3 = weekly_close + range_hl * 1.1 / 4
    weekly_s3 = weekly_close - range_hl * 1.1 / 4
    
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Volume confirmation (20-period MA on 12h = ~10 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA50 (50), weekly pivot (1), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly R3 + volume
            if uptrend and close[i] > weekly_r3_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly S3 + volume
            elif downtrend and close[i] < weekly_s3_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R3
            if not uptrend or close[i] < weekly_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S3
            if not downtrend or close[i] > weekly_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals