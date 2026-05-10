#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_1dTrend_Volume
# Hypothesis: Weekly pivot levels (PP, R1-R4, S1-S4) act as strong support/resistance.
# In trending markets, price breaking above weekly R1 in uptrend or below S1 in downtrend
# continues with momentum. We use 1d EMA34 for trend filter and volume confirmation
# to avoid false breakouts. Works in bull markets (follows uptrends) and bear
# markets (follows downtrends) by only trading in direction of 1d trend.
# Target: 15-35 trades/year to minimize fee drag on 6h timeframe.

name = "6h_Weekly_Pivot_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot points from previous week
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = (2 * PP) - Low
    # S1 = (2 * PP) - High
    # R2 = PP + (High - Low)
    # S2 = PP - (High - Low)
    # R3 = High + 2*(PP - Low)
    # S3 = Low - 2*(High - PP)
    # R4 = R3 + (High - Low)
    # S4 = S3 - (High - Low)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pp = (high_w + low_w + close_w) / 3
    r1 = (2 * pp) - low_w
    s1 = (2 * pp) - high_w
    r2 = pp + (high_w - low_w)
    s2 = pp - (high_w - low_w)
    r3 = high_w + 2 * (pp - low_w)
    s3 = low_w - 2 * (high_w - pp)
    r4 = r3 + (high_w - low_w)
    s4 = s3 - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume confirmation (24-period MA on 6h = ~6 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 (34), weekly pivot (5), volume MA (24)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly R1 + volume
            if uptrend and close[i] > r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly S1 + volume
            elif downtrend and close[i] < s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals