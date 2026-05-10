#!/usr/bin/env python3
"""
1D_Camarilla_R3_S3_Bounce_1wTrend_Volume
Hypothesis: Uses weekly (1w) trend direction via EMA20 and bounce off daily Camarilla R3/S3 levels with volume confirmation.
Designed for 1d timeframe to capture swing reversals in both bull and bear markets by following weekly trend.
Only takes long when price bounces off S3 in weekly uptrend, short when price bounces off R3 in weekly downtrend.
Uses discrete position sizing (0.25) to minimize fee churn and avoid overtrading.
"""

name = "1D_Camarilla_R3_S3_Bounce_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend direction
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for Camarilla pivot calculation (same day's OHLC for intraday levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from current day's OHLC (for intraday bounce)
    # R3 = C + (H-L) * 1.1
    # S3 = C - (H-L) * 1.1
    camarilla_r3 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1
    camarilla_s3 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1
    
    # Align Camarilla levels to 1d timeframe (same bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly trend direction
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        weekly_downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Long entry: price touches/bounces off S3 + weekly uptrend + volume spike
            if (low[i] <= s3_aligned[i] and 
                close[i] > s3_aligned[i] and  # confirmation of bounce
                weekly_uptrend and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price touches/bounces off R3 + weekly downtrend + volume spike
            elif (high[i] >= r3_aligned[i] and 
                  close[i] < r3_aligned[i] and  # confirmation of bounce
                  weekly_downtrend and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or weekly trend turns down
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_1w_aligned[i]):  # trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or weekly trend turns up
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_1w_aligned[i]):  # trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals