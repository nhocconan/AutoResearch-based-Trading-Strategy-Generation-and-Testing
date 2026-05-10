#!/usr/bin/env python3
# 4H_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: Trade in direction of Kaufman Adaptive Moving Average (KAMA) trend on 4h with volume confirmation and daily trend filter.
# KAMA adapts to market noise - follows price closely in trends, stays flat in ranges.
# Long when: KAMA rising, price above KAMA, volume > 1.5x average, daily uptrend.
# Short when: KAMA falling, price below KAMA, volume > 1.5x average, daily downtrend.
# Uses volatility-based adaptation to reduce whipsaws in ranging markets.
# Target: 20-40 trades/year per symbol.

name = "4H_KAMA_Trend_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    close_s = pd.Series(close)
    direction = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # placeholder for efficiency
    # Simplified volatility calculation for 10-period
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, direction / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: KAMA rising, price above KAMA, volume confirmation, daily uptrend
            if kama[i] > kama[i-1] and close[i] > kama[i] and volume_confirm and daily_up:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, price below KAMA, volume confirmation, daily downtrend
            elif kama[i] < kama[i-1] and close[i] < kama[i] and volume_confirm and daily_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling or price below KAMA
            if kama[i] < kama[i-1] or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising or price above KAMA
            if kama[i] > kama[i-1] or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals