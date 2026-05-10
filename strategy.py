#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume
Hypothesis: Price breaks Camarilla R3/S3 levels calculated from 4h data, with 4h EMA50 trend filter and 1d volume spike confirmation.
Camarilla levels provide precise intraday support/resistance; breakouts in the direction of 4h trend capture momentum.
1d volume filter ensures breakouts occur with institutional participation, reducing false signals.
Works in bull/bear by trading only in direction of 4h trend. Target: 20-40 trades/year (80-160 total) to minimize fee drag.
"""

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # 4h data for Camarilla levels and trend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R3 = close + (high-low)*1.1/2, S3 = close - (high-low)*1.1/2
    # Using previous 4h bar's high/low/close for current levels
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    hl_range = prev_high_4h - prev_low_4h
    R3 = prev_close_4h + hl_range * 1.1 / 2
    S3 = prev_close_4h - hl_range * 1.1 / 2
    
    # 4h EMA50 for trend filter
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema50_4h[i-1]
    
    # 1d volume for volume confirmation (average volume)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_avg_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align 4h indicators to 1h
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Align 1d volume to 1h
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 1d volume (scaled)
        # 1d = 24 x 1h, so scale down
        vol_1d_scaled = vol_avg_1d_aligned[i] / 24.0  # Average 1h-equivalent volume from 1d data
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema50_4h_aligned[i]
        is_downtrend = close[i] < ema50_4h_aligned[i]
        price_above_R3 = close[i] > R3_aligned[i]
        price_below_S3 = close[i] < S3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, in uptrend, with volume
            if price_above_R3 and is_uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3, in downtrend, with volume
            elif price_below_S3 and is_downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price falls back below R3 or trend turns down
            if not price_above_R3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price rises back above S3 or trend turns up
            if not price_below_S3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals