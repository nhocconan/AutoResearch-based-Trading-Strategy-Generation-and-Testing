# NOTE: This is a modified version of the original strategy.py
#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_Volume
# Hypothesis: Use weekly pivot point bias and daily trend for direction, enter on 6h pullback with volume.
# Long when weekly close > weekly pivot AND daily close > daily EMA50; enter on 6h pullback to EMA20 with volume.
# Short when weekly close < weekly pivot AND daily close < daily EMA50; enter on 6h bounce to EMA20 with volume.
# Weekly pivot provides structural bias, daily trend confirms, volume confirms momentum.
# Designed for low trade frequency (15-30/year) to avoid fee drag, works in bull/bear via trend filter.

name = "6h_WeeklyPivot_DailyTrend_Volume"
timeframe = "6h"
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
    
    # Weekly data for pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot point calculation (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Bias: price above/below pivot
    weekly_bias_up = close_1w > pivot_1w
    weekly_bias_down = close_1w < pivot_1w
    
    # Daily EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_trend_up = close_1d > ema50_1d
    daily_trend_down = close_1d < ema50_1d
    
    # Align weekly bias to 6h
    weekly_bias_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_up.astype(float))
    weekly_bias_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_down.astype(float))
    
    # Align daily trend to 6h
    daily_trend_up_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up.astype(float))
    daily_trend_down_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_down.astype(float))
    
    # 6h EMA20 for entry timing
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_bias_up_aligned[i]) or np.isnan(weekly_bias_down_aligned[i]) or
            np.isnan(daily_trend_up_aligned[i]) or np.isnan(daily_trend_down_aligned[i]) or
            np.isnan(ema20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: weekly bullish bias + daily uptrend + price near EMA20 with volume
            if (weekly_bias_up_aligned[i] > 0.5 and 
                daily_trend_up_aligned[i] > 0.5 and
                close[i] <= ema20[i] * 1.01 and  # within 1% above EMA20 (pullback)
                close[i] >= ema20[i] * 0.99 and  # within 1% below EMA20
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: weekly bearish bias + daily downtrend + price near EMA20 with volume
            elif (weekly_bias_down_aligned[i] > 0.5 and 
                  daily_trend_down_aligned[i] > 0.5 and
                  close[i] >= ema20[i] * 0.99 and  # within 1% below EMA20
                  close[i] <= ema20[i] * 1.01 and  # within 1% above EMA20
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly bias fails or daily trend fails
            if (weekly_bias_up_aligned[i] < 0.5 or 
                daily_trend_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly bias fails or daily trend fails
            if (weekly_bias_down_aligned[i] < 0.5 or 
                daily_trend_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals