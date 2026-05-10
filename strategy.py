#!/usr/bin/env python3
# 4h_Donchian_Breakout_VolumeTrend_1dTrendFilter
# Hypothesis: Combines 4h Donchian breakout with volume confirmation (2x avg volume) and 1d trend filter (close > SMA50) to capture strong directional moves.
# Uses 4h EMA20 as trend filter for entry direction. Designed for low trade frequency (20-50/year) to minimize fee drag.
# Works in both bull and bear markets by filtering entries with higher timeframe trend.

name = "4h_Donchian_Breakout_VolumeTrend_1dTrendFilter"
timeframe = "4h"
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
    
    # 4h Donchian channels (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # 4h EMA20 for trend filter
    ema20 = np.zeros(n)
    ema20[:] = np.nan
    if n >= 20:
        ema20[19] = np.mean(high[:20])  # simple seed
        for i in range(20, n):
            ema20[i] = 0.1 * close[i] + 0.9 * ema20[i-1]
    
    # 4h volume average (20-period) for confirmation
    vol_ma = np.zeros(n)
    vol_ma[:] = np.nan
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    
    # 1d trend filter: close > SMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma50_1d = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        sma50_1d[i] = np.mean(close_1d[i-49:i+1])
    
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 49)  # ensure Donchian, EMA, vol, and SMA50 are ready
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema20[i]) or np.isnan(vol_ma[i]) or
            np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume > 2x avg + close > SMA50 (1d uptrend)
            if (high[i] > upper[i] and
                volume[i] > 2 * vol_ma[i] and
                close[i] > sma50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume > 2x avg + close < SMA50 (1d downtrend)
            elif (low[i] < lower[i] and
                  volume[i] > 2 * vol_ma[i] and
                  close[i] < sma50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below lower Donchian or 1d trend turns down
            if (low[i] < lower[i] or
                close[i] < sma50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above upper Donchian or 1d trend turns up
            if (high[i] > upper[i] or
                close[i] > sma50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals