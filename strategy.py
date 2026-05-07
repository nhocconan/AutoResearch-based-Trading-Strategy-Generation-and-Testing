#!/usr/bin/env python3
# 4h_KAMA_Trend_Volume_Confirmation
# Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) to capture adaptive trend direction on 4h timeframe,
# confirmed by volume spike and filtered by 1-day trend (EMA34). The strategy adapts to changing market conditions
# (trending vs ranging) via KAMA's efficiency ratio, reducing whipsaws in choppy markets. Volume confirmation
# ensures momentum behind moves. Designed for 4h timeframe to balance trade frequency and signal quality.
# Target: 20-40 trades/year per symbol to stay within optimal range.

timeframe = "4h"
name = "4h_KAMA_Trend_Volume_Confirmation"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 4h close
    # Efficiency Ratio (ER) = |change over period| / sum of absolute changes
    change = np.abs(np.diff(close, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute changes
    # Handle first element
    change = np.concatenate([[np.nan], change])
    vol = np.concatenate([[np.nan], vol])
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2.0x average volume (20-period = ~1.33 days on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume, and 1d trend is bullish (close > EMA34)
            if (close[i] > kama[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume, and 1d trend is bearish (close < EMA34)
            elif (close[i] < kama[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA (trend change)
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA (trend change)
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals