#!/usr/bin/env python3
"""
4h_VolumeBreakout_1dTrend_WeeklyTrend
Hypothesis: Volume-confirmed breakout of 4h range with 1d and 1w trend filters.
Long when price breaks above 4h high with volume > 2x average and both 1d/1w trends up.
Short when price breaks below 4h low with volume > 2x average and both 1d/1w trends down.
Exit when price returns to 4h mid-range or trend reverses.
Designed to capture strong momentum moves while avoiding false breakouts in low volume.
Target: 20-35 trades/year to minimize fee drag while capturing major moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        multiplier = 2 / (ema_1d_period + 1)
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_1w_period:
        ema_1w[ema_1w_period - 1] = np.mean(close_1w[:ema_1w_period])
        multiplier = 2 / (ema_1w_period + 1)
        for i in range(ema_1w_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Calculate 4h range (20-period high/low)
    high_max_20 = np.full(n, np.nan)
    low_min_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_max_20[i] = np.max(high[i-20:i])
        low_min_20[i] = np.min(low[i-20:i])
    
    # Calculate 4h volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align all indicators to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 50)  # 4h range needs 20, EMA needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(high_max_20[i]) or
            np.isnan(low_min_20[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Trend filters: both 1d and 1w EMA50 must agree
        uptrend = price > ema_1d_aligned[i] and price > ema_1w_aligned[i]
        downtrend = price < ema_1d_aligned[i] and price < ema_1w_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above 4h high with volume and uptrend
            if uptrend and volume_confirmation and price > high_max_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 4h low with volume and downtrend
            elif downtrend and volume_confirmation and price < low_min_20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: return to 4h mid-range or trend reversal
            mid_range = (high_max_20[i] + low_min_20[i]) / 2
            if price < mid_range or not (price > ema_1d_aligned[i] and price > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: return to 4h mid-range or trend reversal
            mid_range = (high_max_20[i] + low_min_20[i]) / 2
            if price > mid_range or not (price < ema_1d_aligned[i] and price < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_VolumeBreakout_1dTrend_WeeklyTrend"
timeframe = "4h"
leverage = 1.0