#!/usr/bin/env python3
"""
1d_WeeklyPivot_Donchian_Breakout_Trend
Hypothesis: On daily timeframe, price breaks Donchian(20) high/low when aligned with weekly EMA200 trend and confirmed by volume spike. Weekly EMA200 provides strong trend filter for bull/bear markets, Donchian breakouts capture momentum, and volume confirmation reduces false signals. Designed for low trade frequency (target: 15-25/year) to minimize fee drag and work in both bull and bear regimes.
"""

name = "1d_WeeklyPivot_Donchian_Breakout_Trend"
timeframe = "1d"
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
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on weekly close
    ema_200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_200_1w[i-1]
    
    # Align weekly EMA200 to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily volume SMA20 for volume confirmation
    volume_sma_20 = np.full(n, np.nan)
    if n >= 20:
        volume_sma_20[19] = np.mean(volume[:20])
        for i in range(20, n):
            volume_sma_20[i] = (volume_sma_20[i-1] * 19 + volume[i]) / 20
    
    # Donchian channels (20-period) on daily high/low
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for weekly EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_sma_20[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-day average
        volume_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        if position == 0:
            # Long: Break above Donchian high with uptrend and volume
            if close[i] > donchian_high[i] and close[i] > ema_200_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with downtrend and volume
            elif close[i] < donchian_low[i] and close[i] < ema_200_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below Donchian low (trend reversal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above Donchian high (trend reversal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals