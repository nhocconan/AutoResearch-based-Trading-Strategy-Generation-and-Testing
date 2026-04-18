#!/usr/bin/env python3
"""
1d_KAMA_Trend_WeeklyTrendFilter
Hypothesis: Follow the weekly trend using KAMA on 1d for entry/exit. KAMA adapts to market efficiency, reducing whipsaw in sideways markets. Go long when weekly EMA(34) is rising and 1d KAMA turns up; short when weekly EMA(34) is falling and 1d KAMA turns down. Use volume > 1.5x 20-day average for confirmation. Designed for low trade frequency (~10-20 trades/year) to minimize fee drag and work in both bull and bear markets by following the higher-timeframe trend.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend direction
    ema_period = 34
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (ema_period + 1)) + (ema_1w[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get daily data for KAMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)   # 2-period EMA smoothing constant
    slow_sc = 2 / (30 + 1)  # 30-period EMA smoothing constant
    
    kama = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 1:
        kama[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            # Efficiency ratio
            change = abs(close_1d[i] - close_1d[i-1])
            volatility = 0
            for j in range(1, i+1):
                volatility += abs(close_1d[j] - close_1d[j-1])
            er = change / volatility if volatility != 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily (trivial since same timeframe, but using alignment for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period, ema_period)  # KAMA needs ~30, vol MA 20, EMA 34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or i == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Weekly trend: rising if today's EMA > yesterday's EMA
        weekly_rising = ema_1w_aligned[i] > ema_1w_aligned[i-1]
        weekly_falling = ema_1w_aligned[i] < ema_1w_aligned[i-1]
        
        # KAMA direction: turning up/down
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        if position == 0:
            # Long: weekly trend up + KAMA turns up + volume
            if weekly_rising and kama_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: weekly trend down + KAMA turns down + volume
            elif weekly_falling and kama_down and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns down or KAMA turns down
            if weekly_falling or kama_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up or KAMA turns up
            if weekly_rising or kama_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0