#!/usr/bin/env python3
# 6H_1W_1D_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume
# Hypothesis: On 6h timeframe, enter long when price breaks above Camarilla R3 level from previous 1d candle with 1w uptrend and volume confirmation.
# Short when price breaks below Camarilla S3 level with 1w downtrend and volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades and Camarilla levels from 1d for precise entries.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "6H_1W_1D_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: R3, S3 based on previous day
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly trend: EMA(34) on close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = close_1w > ema_34_1w
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 1d and 1w indicators to 6h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(trend_up_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R3 + 1w uptrend + volume confirmation
            if close[i] > camarilla_r3_aligned[i] and trend_up_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S3 + 1w downtrend + volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and not trend_up_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 (reversal) or trend changes
            if close[i] < camarilla_s3_aligned[i] or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 (reversal) or trend changes
            if close[i] > camarilla_r3_aligned[i] or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals