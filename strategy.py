#!/usr/bin/env python3
"""
1d_Donchian20_TrendFilter_VolumeConfirm
Hypothesis: Breakout of 20-day Donchian channel (highest high/lowest low) with weekly EMA50 trend filter and daily volume confirmation.
Works in bull markets via upward breakouts; works in bear markets via downward breakouts filtered by weekly trend.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "1d_Donchian20_TrendFilter_VolumeConfirm"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Daily Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Daily volume SMA20 for confirmation
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    # Align weekly EMA50 to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA50 and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        # Trend filter: price above/below weekly EMA50
        is_uptrend = close[i] > ema50_1w_aligned[i]
        is_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Donchian breakout
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        if position == 0:
            # Long: upward breakout in uptrend with volume
            if breakout_up and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout in downtrend with volume
            elif breakout_down and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below Donchian lower band or trend turns down
            if close[i] < lowest_low[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above Donchian upper band or trend turns up
            if close[i] > highest_high[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals