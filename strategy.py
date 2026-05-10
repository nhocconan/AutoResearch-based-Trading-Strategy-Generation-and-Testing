#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume
Hypothesis: Price breaks Donchian(20) channel calculated from prior 12h session, with confirmation from 1d EMA50 trend and volume spike. Donchian channels provide clear breakout signals in trending markets, while EMA50 trend filter ensures alignment with higher timeframe direction. Volume confirmation reduces false breakouts. Target: 12-37 trades/year (50-150 total over 4 years).
"""

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 12h data for Donchian channel (using prior completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian(20) channel for each 12h bar
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    donchian_upper_12h = np.full(len(high_12h), np.nan)
    donchian_lower_12h = np.full(len(low_12h), np.nan)
    
    if len(high_12h) >= 20:
        for i in range(20, len(high_12h)):
            donchian_upper_12h[i] = np.max(high_12h[i-20:i])
            donchian_lower_12h[i] = np.min(low_12h[i-20:i])
        # Initialize first 20 values with NaN (not enough data)
    
    # Align Donchian levels to 12h timeframe (wait for 12h bar to close)
    donchian_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
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
           np.isnan(donchian_upper_12h_aligned[i]) or np.isnan(donchian_lower_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled to 12h)
        # Approximate 12h volume from 1d: 1d volume / 2 (since 24h/12h = 2)
        vol_12h_approx = vol_sma_20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Break above Donchian upper with uptrend and volume
            if close[i] > donchian_upper_12h_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower with downtrend and volume
            elif close[i] < donchian_lower_12h_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
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