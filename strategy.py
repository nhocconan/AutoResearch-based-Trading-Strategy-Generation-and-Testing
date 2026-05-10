#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSurge
# Hypothesis: Camarilla pivot levels on daily chart provide strong support/resistance.
# Breakout above R1 or below S1 with volume surge confirms institutional interest.
# 1-day EMA50 filter ensures we trade in direction of higher timeframe trend.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Volume surge filter avoids false breakouts in low-liquidity periods.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSurge"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous daily candle
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero and handle NaN
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.1 / 12
    s1 = prev_close - rang * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: 24-period MA on 12h chart = 12 days
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50_1d (50), volume MA (24)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume surge confirmation (at least 1.5x average volume)
        volume_surge = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 in uptrend with volume surge
            if uptrend and close[i] > r1_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 in downtrend with volume surge
            elif downtrend and close[i] < s1_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend breaks
            if close[i] < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend breaks
            if close[i] > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals