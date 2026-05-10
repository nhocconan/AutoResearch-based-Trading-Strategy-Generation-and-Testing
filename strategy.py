#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Daily chart Camarilla R3/S3 breakout with weekly trend filter and volume spike.
Weekly trend (SMA200) filters direction to avoid counter-trend trades. Volume confirmation ensures
institutional participation. Targets 10-25 trades/year on 1d timeframe to minimize fee drag.
Works in bull/bear markets by using weekly trend to select direction and Camarilla levels for
low-risk breakout entries.
"""
name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly SMA200 for trend filter (requires 200 weeks)
    close_1w = df_1w['close'].values
    sma_200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    sma_200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    # Camarilla R3 and S3
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.25
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.25
    
    # Align Camarilla levels to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly SMA200 (200)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sma_200_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with weekly uptrend and volume
            if close[i] > camarilla_r3_aligned[i] and close[i] > sma_200_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with weekly downtrend and volume
            elif close[i] < camarilla_s3_aligned[i] and close[i] < sma_200_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price back below S3 or weekly trend turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < sma_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price back above R3 or weekly trend turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > sma_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals