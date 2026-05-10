#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Filter
Hypothesis: Price breaks above/below weekly Camarilla R3/S3 levels on daily timeframe with weekly trend filter and volume confirmation.
Targets major weekly support/resistance levels to capture significant moves while avoiding whipsaws.
Designed for low trade frequency (10-25 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Filter"
timeframe = "1d"
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
    
    # Weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Camarilla pivot levels (R3/S3)
    range_1w = high_1w - low_1w
    r3_1w = close_1w + 1.1 * range_1w
    s3_1w = close_1w - 1.1 * range_1w
    
    # Weekly EMA20 for trend filter
    close_1w_ema = df_1w['close'].values
    ema20_1w = np.full(len(close_1w_ema), np.nan)
    if len(close_1w_ema) >= 20:
        ema20_1w[19] = np.mean(close_1w_ema[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w_ema)):
            ema20_1w[i] = alpha * close_1w_ema[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Daily volume SMA20 for volume confirmation
    volume_1d = prices['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align all indicators to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    vol_sma20_1d_aligned = vol_sma20_1d  # already daily
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA20
    
    for i in range(start_idx, n):
        if np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * vol_sma20_1d_aligned[i]
        
        # Trend and price relative to Weekly Camarilla levels
        is_uptrend = close[i] > ema20_1w_aligned[i]
        is_downtrend = close[i] < ema20_1w_aligned[i]
        price_above_r3 = close[i] > r3_1w_aligned[i]
        price_below_s3 = close[i] < s3_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R3, in weekly uptrend, with volume
            if price_above_r3 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, in weekly downtrend, with volume
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