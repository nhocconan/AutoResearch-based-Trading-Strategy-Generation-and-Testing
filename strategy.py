#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend
Hypothesis: Price breaks Camarilla R3 (long) or S3 (short) levels calculated from prior week's range, with 1w EMA50 trend filter and volume confirmation. Uses 12h timeframe to reduce trade frequency and focus on weekly structure. Works in bull/bear by filtering trades in direction of weekly trend. Target: 20-30 trades/year (80-120 total) to minimize fee drag.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend"
timeframe = "12h"
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
    
    # 1w data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels from prior week: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w) / 4
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w) / 4
    
    # 1w EMA50 for trend filter
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # 1w volume SMA10 for volume confirmation
    vol_sma10_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 10:
        vol_sma10_1w[9] = np.mean(df_1w['volume'].values[:10])
        for i in range(10, len(df_1w)):
            vol_sma10_1w[i] = (vol_sma10_1w[i-1] * 9 + df_1w['volume'].values[i]) / 10
    
    # Align 1w indicators to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma10_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1w volume (scaled)
        vol_1w_scaled = vol_sma10_1w_aligned[i] / 14.0  # 14x 12h bars in 1w
        volume_confirm = volume[i] > 1.5 * vol_1w_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema50_1w_aligned[i]
        is_downtrend = close[i] < ema50_1w_aligned[i]
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