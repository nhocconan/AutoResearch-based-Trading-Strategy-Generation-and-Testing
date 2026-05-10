#!/usr/bin/env python3
# 1D_Donchian_Breakout_WeeklyTrend_Volume
# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above upper Donchian + weekly uptrend + volume > 1.5x average.
# Short when price breaks below lower Donchian + weekly downtrend + volume > 1.5x average.
# Exit when price closes back inside the Donchian channel.
# Designed for low trade frequency (~15-25/year) to avoid fee drag, works in bull/bear by following weekly trend.

name = "1D_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(20, n):
        highest[i] = np.max(high[i-20:i])
        lowest[i] = np.min(low[i-20:i])
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        trend_up = trend_1w_up_aligned[i] > 0.5
        trend_down = trend_1w_down_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: breakout above upper Donchian + weekly uptrend + volume
            if close[i] > highest[i] and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below lower Donchian + weekly downtrend + volume
            elif close[i] < lowest[i] and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price closes back inside Donchian channel
            if close[i] < highest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes back inside Donchian channel
            if close[i] > lowest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals