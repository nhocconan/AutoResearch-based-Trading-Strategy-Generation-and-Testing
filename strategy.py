#!/usr/bin/env python3
# 4H_1D_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: On 4h timeframe, enter long when KAMA direction is up and price crosses above 4h EMA(20) with volume confirmation.
# Short when KAMA direction is down and price crosses below 4h EMA(20) with volume confirmation.
# Uses 1d KAMA trend filter to avoid counter-trend trades and ensure alignment with higher timeframe momentum.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).

name = "4H_1D_KAMA_Trend_With_Volume_Confirmation"
timeframe = "4h"
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
    
    # Get 1d data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d))[1:]) if len(close_1d) > 1 else 1
    er = np.zeros_like(close_1d)
    er[1:] = np.abs(np.diff(close_1d)) / (np.sum(np.abs(np.diff(close_1d))) + 1e-10)
    # Smooth ER with smoothing constants
    sc = (er * 0.0645 + 0.0645) ** 2  # where 0.0645 = 2/(2+1) for fast EMA(2)
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 1d trend: price above KAMA
    trend_up = close_1d > kama
    
    # 4h EMA(20) for entry timing
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 1d indicators to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trend_up_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA uptrend + price crosses above EMA(20) + volume confirmation
            if trend_up_aligned[i] and close[i] > ema_20[i] and close[i-1] <= ema_20[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA downtrend + price crosses below EMA(20) + volume confirmation
            elif not trend_up_aligned[i] and close[i] < ema_20[i] and close[i-1] >= ema_20[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA(20) or trend changes
            if close[i] < ema_20[i] and close[i-1] >= ema_20[i-1] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA(20) or trend changes
            if close[i] > ema_20[i] and close[i-1] <= ema_20[i-1] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals