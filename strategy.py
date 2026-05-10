#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Price breaks Camarilla R3 (long) or S3 (short) levels calculated from prior daily session, with confirmation from 1w EMA34 trend and volume spike. Camarilla R3/S3 represent stronger breakout levels than R1/S1, reducing false signals. The 1w trend filter ensures alignment with higher timeframe direction, and volume confirmation reduces noise. Target: 15-30 trades/year.
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
    
    # Daily data for Camarilla levels (using prior completed daily bar)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar: R3, S3
    # R3 = close + (high - low) * 1.25
    # S3 = close - (high - low) * 1.25
    camarilla_r3_1d = close_1d + (high_1d - low_1d) * 1.25
    camarilla_s3_1d = close_1d - (high_1d - low_1d) * 1.25
    
    # Align Camarilla levels to 12h timeframe (wait for daily bar to close)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_34_1w[i-1]
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or \
           np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average daily volume (scaled to 12h)
        # Approximate 12h volume from daily: daily volume / 2 (since 24h/12h = 2)
        vol_12h_approx = vol_sma_20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Break above Camarilla R3 with uptrend and volume
            if close[i] > camarilla_r3_12h[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 with downtrend and volume
            elif close[i] < camarilla_s3_12h[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA34 (trend reversal)
            if close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA34 (trend reversal)
            if close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals