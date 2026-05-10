#!/usr/bin/env python3
# 1d_EMA_Cross_Trend_With_Volume_Confirmation
# Hypothesis: EMA crossovers on daily timeframe capture medium-term trends, and volume confirmation filters false signals.
# In bull markets, EMA(21) crossing above EMA(50) signals uptrend continuation; in bear markets, crossing below signals downtrend.
# Volume > 1.5x 20-period average confirms institutional participation. Works in both bull and bear by following trend direction.

name = "1d_EMA_Cross_Trend_With_Volume_Confirmation"
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
    
    # Calculate EMA21 and EMA50 on daily data
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period moving average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_21[i]) or 
            np.isnan(ema_50[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # EMA crossover signals
        ema_21_above_50 = ema_21[i] > ema_50[i]
        ema_21_below_50 = ema_21[i] < ema_50[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: EMA21 crosses above EMA50 + volume confirmation
            if ema_21_above_50 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: EMA21 crosses below EMA50 + volume confirmation
            elif ema_21_below_50 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: EMA21 crosses below EMA50
            if ema_21_below_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: EMA21 crosses above EMA50
            if ema_21_above_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals