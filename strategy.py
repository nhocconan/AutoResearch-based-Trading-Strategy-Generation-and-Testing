#!/usr/bin/env python3
"""
1d_WeeklyPivot_Donchian_Breakout_Trend_1w
Hypothesis: Price breaks Donchian(20) high (long) or low (short) calculated from prior 1d candles, with confirmation from weekly EMA50 trend and volume spike. Weekly pivot provides structural levels, Donchian breakout captures momentum, EMA50 trend filter ensures alignment with higher timeframe direction. Volume confirmation reduces false breakouts. Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "1d_WeeklyPivot_Donchian_Breakout_Trend_1w"
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
    
    # Weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly volume SMA20 for volume confirmation
    volume_1w = df_1w['volume'].values
    vol_sma_20_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_sma_20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_sma_20_1w[i] = (vol_sma_20_1w[i-1] * 19 + volume_1w[i]) / 20
    vol_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma_20_1w)
    
    # Daily Donchian channels (20-period) - calculated from prior completed daily candles
    # We need at least 20 days of data to calculate Donchian channels
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20_1w_aligned[i]) or \
           np.isnan(highest_20[i]) or np.isnan(lowest_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume approximation: daily volume from weekly (weekly / 5)
        vol_daily_approx = vol_sma_20_1w_aligned[i] / 5.0
        volume_confirm = volume[i] > 1.5 * vol_daily_approx
        
        if position == 0:
            # Long: Break above Donchian high with uptrend and volume
            if close[i] > highest_20[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with downtrend and volume
            elif close[i] < lowest_20[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
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