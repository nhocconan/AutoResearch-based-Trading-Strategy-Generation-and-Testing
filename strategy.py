#!/usr/bin/env python3
"""
1d_WeeklyPivot_Donchian_Breakout_Trend_1w
Hypothesis: Daily price breaks above/below weekly Donchian channel (20) in direction of weekly EMA20 trend, with volume confirmation from daily volume > 1.5x 20-day average. 
Weekly trend filter ensures alignment with longer-term momentum, working in both bull and bear markets. 
Volume confirmation reduces false breakouts. Target: 10-25 trades/year.
"""

name = "1d_WeeklyPivot_Donchian_Breakout_Trend_1w"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly Donchian channel (20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20 = np.full(len(high_1w), np.nan)
    donchian_low_20 = np.full(len(low_1w), np.nan)
    if len(high_1w) >= 20:
        for i in range(20, len(high_1w)):
            donchian_high_20[i] = np.max(high_1w[i-20:i])
            donchian_low_20[i] = np.min(low_1w[i-20:i])
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Daily volume SMA (20) for confirmation
    vol_sma_20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma_20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma_20[i] = (vol_sma_20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian + volume warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or np.isnan(vol_sma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: daily volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * vol_sma_20[i]
        
        if position == 0:
            # Long: Break above weekly Donchian high and above weekly EMA20
            if close[i] > donchian_high_20_aligned[i] and close[i] > ema_20_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian low and below weekly EMA20
            elif close[i] < donchian_low_20_aligned[i] and close[i] < ema_20_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly EMA20
            if close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly EMA20
            if close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals