#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend
Hypothesis: Price breaks Camarilla R3 (long) or S3 (short) levels calculated from prior day's range, with 1w EMA200 trend filter and volume confirmation.
R3/S3 represent stronger support/resistance than R1/S1, reducing false breakouts.
Weekly trend filter ensures trades align with higher timeframe momentum.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
Works in bull/bear by filtering trades in direction of weekly trend.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Camarilla levels from prior day: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # 1w EMA200 for trend filter
    ema200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema200_1w[i-1]
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(vol_1d), np.nan)
    if len(vol_1d) >= 20:
        vol_sma20_1d[19] = np.mean(vol_1d[:20])
        for i in range(20, len(vol_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + vol_1d[i]) / 20
    
    # Align 1d and 1w indicators to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 2.0  # 2x 12h bars in 1d
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema200_1w_aligned[i]
        is_downtrend = close[i] < ema200_1w_aligned[i]
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, in uptrend, with volume
            if price_above_r3 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, in downtrend, with volume
            elif price_below_s3 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R3 or trend turns down
            if not price_above_r3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S3 or trend turns up
            if not price_below_s3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals