#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla R3/S3 levels from 1d act as strong support/resistance in trending markets.
In trending markets (determined by 1w EMA34), price tends to break through R3/S3 with momentum.
Long when price breaks above R3 in uptrend with volume confirmation.
Short when price breaks below S3 in downtrend with volume confirmation.
Uses 12h timeframe for lower frequency (target: 50-150 trades over 4 years) to minimize fee drag.
Works in both bull (breakouts above resistance) and bear (breakdowns below support).
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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1d volume SMA20 for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    camarilla_width = (high_1d - low_1d) * 1.1
    r3_1d = close_1d + camarilla_width / 4.0
    s3_1d = close_1d - camarilla_width / 4.0
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # Need EMA34 and at least one Camarilla level
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled to 12h)
        vol_12h_approx = vol_sma20_1d_aligned[i] / 2.0  # 2x 12h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Price breaks above R3 in uptrend with volume confirmation
            if close[i] > r3_1d_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 in downtrend with volume confirmation
            elif close[i] < s3_1d_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls back below R3 or trend reversal
            if close[i] < r3_1d_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above S3 or trend reversal
            if close[i] > s3_1d_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals