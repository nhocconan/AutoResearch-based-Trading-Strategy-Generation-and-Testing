#!/usr/bin/env python3
"""
1d_WeeklyPivot_Donchian_Breakout_Trend
Hypothesis: On daily timeframe, price breaks Donchian(20) high/low with confirmation from weekly trend (EMA21) and volume surge. Weekly pivot levels act as support/resistance. This captures breakouts in both bull and bear markets while filtering false signals with trend and volume. Target: 10-25 trades/year.
"""

name = "1d_WeeklyPivot_Donchian_Breakout_Trend"
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
    
    # Weekly data for trend filter (EMA21)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        ema_21_1w[20] = np.mean(close_1w[:21])
        alpha = 2 / (21 + 1)
        for i in range(21, len(close_1w)):
            ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily volume SMA20 for volume confirmation
    volume_sma_20 = np.full(n, np.nan)
    if n >= 20:
        volume_sma_20[19] = np.mean(volume[:20])
        for i in range(20, n):
            volume_sma_20[i] = (volume_sma_20[i-1] * 19 + volume[i]) / 20
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_sma_20[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        if position == 0:
            # Long: Break above Donchian high with uptrend and volume
            if close[i] > donchian_high[i] and close[i] > ema_21_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with downtrend and volume
            elif close[i] < donchian_low[i] and close[i] < ema_21_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA21 (trend reversal) or Donchian low (mean reversion)
            if close[i] < ema_21_1w_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA21 (trend reversal) or Donchian high (mean reversion)
            if close[i] > ema_21_1w_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals