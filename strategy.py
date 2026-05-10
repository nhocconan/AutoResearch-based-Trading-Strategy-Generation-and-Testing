#!/usr/bin/env python3
# 4H_1D_1W_Camarilla_R1_S1_Breakout_TrendFilter
# Hypothesis: Use Camarilla pivot levels from 1d to identify key S/R zones and 1w trend filter to avoid counter-trend trades.
# Enter on breakout of R1 (long) or S1 (short) with volume confirmation. Exit at opposite Camarilla level (S1 for long, R1 for short).
# Works in bull markets by buying strength and in bear markets by selling weakness.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).

name = "4H_1D_1W_Camarilla_R1_S1_Breakout_TrendFilter"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    r2 = close_1d + (high_1d - low_1d) * 1.1 / 6.0
    s2 = close_1d - (high_1d - low_1d) * 1.1 / 6.0
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple trend: price above/below 20-period EMA
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Trend: 1 = uptrend (price > EMA20), -1 = downtrend (price < EMA20), 0 = unclear
    trend_1w = np.where(close_1w > ema_20_1w, 1, np.where(close_1w < ema_20_1w, -1, 0))
    
    # Align all levels and trend to 4h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend_1w.astype(float))
    
    # Volume filter: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = vol_filter[i]
        
        # Trend filter
        uptrend = trend_aligned[i] > 0.5
        downtrend = trend_aligned[i] < -0.5
        
        if position == 0:
            # Enter long: price breaks above R1 + volume + uptrend or ranging
            if close[i] > r1_aligned[i] and vol_ok and (uptrend or abs(trend_aligned[i]) < 0.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + volume + downtrend or ranging
            elif close[i] < s1_aligned[i] and vol_ok and (downtrend or abs(trend_aligned[i]) < 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (opposite level) or reverse signal
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (opposite level) or reverse signal
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals