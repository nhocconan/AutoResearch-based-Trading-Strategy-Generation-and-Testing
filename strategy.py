#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Price breaks Camarilla R3 or S3 levels calculated from prior weekly close, with confirmation from weekly EMA50 trend and volume spike. Camarilla levels provide high-probability reversal/breakout points in ranging markets, while weekly EMA50 ensures alignment with higher timeframe direction. Volume confirmation reduces false breakouts. Target: 7-25 trades/year (30-100 total over 4 years).
"""

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla levels and EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Camarilla levels for each weekly bar
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r3_1w = np.full(len(close_1w), np.nan)
    camarilla_s3_1w = np.full(len(close_1w), np.nan)
    
    if len(high_1w) >= 1:
        for i in range(len(high_1w)):
            high_low_diff = high_1w[i] - low_1w[i]
            camarilla_r3_1w[i] = close_1w[i] + 1.1 * high_low_diff
            camarilla_s3_1w[i] = close_1w[i] - 1.1 * high_low_diff
    
    # Align Camarilla levels to daily timeframe (wait for weekly bar to close)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly volume SMA20 for volume confirmation
    vol_sma_20_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_sma_20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_sma_20_1w[i] = (vol_sma_20_1w[i-1] * 19 + volume_1w[i]) / 20
    vol_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20_1w_aligned[i]) or \
           np.isnan(camarilla_r3_1w_aligned[i]) or np.isnan(camarilla_s3_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average weekly volume (scaled to daily)
        # Approximate daily volume from weekly: weekly volume / 5 (since 5 trading days per week)
        vol_daily_approx = vol_sma_20_1w_aligned[i] / 5.0
        volume_confirm = volume[i] > 1.5 * vol_daily_approx
        
        if position == 0:
            # Long: Break above Camarilla R3 with uptrend and volume
            if close[i] > camarilla_r3_1w_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 with downtrend and volume
            elif close[i] < camarilla_s3_1w_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA50 (trend reversal)
            if close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA50 (trend reversal)
            if close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals