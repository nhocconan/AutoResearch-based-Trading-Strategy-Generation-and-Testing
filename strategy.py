#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Price breaks Camarilla R3 or S3 level calculated from prior 1d session, with confirmation from 1d EMA50 trend and volume spike. Camarilla levels provide high-probability reversal/breakout points, while EMA50 trend filter ensures alignment with higher timeframe direction. Volume confirmation reduces false breakouts. Target: 12-37 trades/year (50-150 total over 4 years).
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # 1d data for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each 1d bar: R3, S3
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(low_1d), np.nan)
    
    if len(high_1d) >= 1:
        for i in range(len(high_1d)):
            high_val = high_1d[i]
            low_val = low_1d[i]
            close_val = close_1d[i]
            range_val = high_val - low_val
            camarilla_r3[i] = close_val + range_val * 1.1 / 4
            camarilla_s3[i] = close_val - range_val * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume SMA20 for volume confirmation
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume (scaled from 1d)
        vol_12h_approx = vol_sma_20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Break above R3 with uptrend and volume
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend and volume
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA50 (trend reversal)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA50 (trend reversal)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals