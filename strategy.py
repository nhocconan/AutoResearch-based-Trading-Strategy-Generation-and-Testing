#!/usr/bin/env python3
# 1D_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: On daily timeframe, enter long when KAMA turns upward with volume confirmation, short when KAMA turns downward.
# Uses 1-week trend filter to avoid counter-trend trades. Designed for low trade frequency (15-25/year) to minimize fee drag.
# KAUFMAN ADAPTIVE MOVING AVERAGE adapts to market noise, reducing whipsaws in sideways markets.
# Volume confirmation ensures institutional participation. Works in both bull and bear via trend filter.

name = "1D_KAMA_Trend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Fix: volatility needs rolling sum
    volatility_rolling = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_rolling > 0, change / volatility_rolling, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: slope > 0
    kama_slope = np.diff(kama, prepend=0)
    kama_up = kama_slope > 0
    
    # Weekly trend: EMA(34) on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = close_1w > ema_34_1w
    
    # Volume confirmation: current volume > 1.8x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.8)
    
    # Align weekly trend to daily
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(kama_up[i]) or np.isnan(weekly_trend_up_aligned[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA turning up + weekly uptrend + volume confirmation
            if kama_up[i] and weekly_trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA turning down + weekly downtrend + volume confirmation
            elif not kama_up[i] and not weekly_trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down or weekly trend changes
            if not kama_up[i] or not weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up or weekly trend changes
            if kama_up[i] or weekly_trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals