#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Filter
Hypothesis: Price breaks above/below weekly Camarilla R3/S3 levels with daily EMA100 trend filter and volume confirmation.
Weekly levels provide strong support/resistance; daily trend ensures alignment with intermediate momentum.
Volume filters false breakouts. Works in bull/bear by trading only in direction of daily trend.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

name = "1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Filter"
timeframe = "1d"
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
    
    # Weekly data for Camarilla pivot levels (R3/S3)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla R3 and S3 levels
    range_1w = high_1w - low_1w
    r3_1w = close_1w + 1.1 * range_1w
    s3_1w = close_1w - 1.1 * range_1w
    
    # Daily EMA100 for trend filter
    ema100 = np.full(n, np.nan)
    if n >= 100:
        ema100[99] = np.mean(close[:100])
        alpha = 2 / (100 + 1)
        for i in range(100, n):
            ema100[i] = alpha * close[i] + (1 - alpha) * ema100[i-1]
    
    # Daily volume SMA20 for volume confirmation
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    # Align weekly levels to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for EMA100
    
    for i in range(start_idx, n):
        if np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or np.isnan(ema100[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        # Trend and price relative to weekly Camarilla levels
        is_uptrend = close[i] > ema100[i]
        is_downtrend = close[i] < ema100[i]
        price_above_r3 = close[i] > r3_1w_aligned[i]
        price_below_s3 = close[i] < s3_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R3, in uptrend, with volume
            if price_above_r3 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, in downtrend, with volume
            elif price_below_s3 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly R3 or trend turns down
            if not price_above_r3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly S3 or trend turns up
            if not price_below_s3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals