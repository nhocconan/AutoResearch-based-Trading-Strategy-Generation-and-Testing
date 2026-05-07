#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Filter
Hypothesis: KAMA direction + RSI + chop filter works on 1d timeframe by capturing trends while avoiding chop.
Targets 20-40 trades per year. Works in both bull and bear markets via regime filter.
"""
name = "1d_KAMA_Trend_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if i == 0:
            er[i] = 0
        else:
            dir_change = np.abs(close[i] - close[i-10]) if i >= 10 else np.abs(close[i] - close[0])
            sum_vol = np.sum(volatility[max(0, i-9):i+1]) if i >= 9 else np.sum(volatility[:i+1])
            er[i] = dir_change / sum_vol if sum_vol != 0 else 0
    
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA + RSI > 50 + chop < 61.8 (trending) + volume
            if close[i] > kama[i] and rsi[i] > 50 and chop[i] < 61.8 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA + RSI < 50 + chop < 61.8 (trending) + volume
            elif close[i] < kama[i] and rsi[i] < 50 and chop[i] < 61.8 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: opposite conditions
            if position == 1:
                if close[i] < kama[i] or rsi[i] < 50 or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or rsi[i] > 50 or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals