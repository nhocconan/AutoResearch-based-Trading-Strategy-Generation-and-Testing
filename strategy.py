#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Price breaks weekly Camarilla R3 (long) or S3 (short) levels calculated from prior week's range, with 1w EMA34 trend filter and volume confirmation.
Weekly levels provide stronger support/resistance than daily. Trend filter ensures trades align with higher timeframe momentum.
Volume confirmation avoids false breakouts. Designed for 12h timeframe to capture multi-day moves with low frequency.
Target: 15-25 trades/year (60-100 total) to minimize fee drag and work in both bull/bear markets.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Weekly Camarilla levels from prior week: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w) / 4
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w) / 4
    
    # 1w EMA34 for trend filter
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    
    # 1w volume SMA10 for volume confirmation
    vol_sma10_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 10:
        vol_sma10_1w[9] = np.mean(df_1w['volume'].values[:10])
        for i in range(10, len(df_1w)):
            vol_sma10_1w[i] = (vol_sma10_1w[i-1] * 9 + df_1w['volume'].values[i]) / 10
    
    # Align 1w indicators to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    vol_sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma10_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1w volume (scaled)
        vol_1w_scaled = vol_sma10_1w_aligned[i] / 2.0  # 2x 12h bars in 1w
        volume_confirm = volume[i] > 1.5 * vol_1w_scaled
        
        # Trend and price relative to Weekly Camarilla levels
        is_uptrend = close[i] > ema34_1w_aligned[i]
        is_downtrend = close[i] < ema34_1w_aligned[i]
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