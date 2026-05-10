#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Fade_1dTrend_Volume
# Hypothesis: Fade at Camarilla R3/S3 levels on 6h chart with 1-day trend filter and volume confirmation.
# In strong trends (1d EMA50 aligned), price often pulls back to R3/S3 before continuing.
# Uses mean reversion at these levels with tight stops via position sizing (0.25).
# Works in bull/bear by requiring 1d trend alignment, avoiding counter-trend traps.
# Targets 50-150 trades over 4 years via strict entry conditions.

name = "6h_Camarilla_R3_S3_Fade_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLC
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Where C, H, L are from previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3 for each day
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 6h timeframe (wait for day to complete)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for Camarilla (need previous day) + EMA50 + vol MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (1.3x average)
        volume_surge = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Price near S3 in uptrend with volume
            near_s3 = abs(close[i] - s3_aligned[i]) / s3_aligned[i] < 0.002  # within 0.2%
            if near_s3 and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Price near R3 in downtrend with volume
            elif abs(close[i] - r3_aligned[i]) / r3_aligned[i] < 0.002 and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price moves to midpoint or trend fails
                midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
                if close[i] > midpoint or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price moves to midpoint or trend fails
                midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
                if close[i] < midpoint or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals