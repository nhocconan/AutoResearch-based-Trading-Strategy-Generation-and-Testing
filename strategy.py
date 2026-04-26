#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter_v1
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum and Choppiness Index for regime filtering.
Only take longs when KAMA is rising, RSI > 50, and market is trending (CHOP < 38.2).
Only take shorts when KAMA is falling, RSI < 50, and market is trending (CHOP < 38.2).
In choppy markets (CHOP >= 38.2), remain flat to avoid whipsaws.
Designed for low trade frequency (7-25/year) by requiring confluence of trend, momentum, and regime.
Works in both bull and bear markets via KAMA's adaptive trend detection and regime filter.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

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
    
    # === KAMA Calculation (trend direction) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array alignment: volatility needs to be computed properly
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True
    ).values
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA 2
    slow_sc = 2 / (30 + 1)  # for EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed at index 9 (10th element)
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros(n)
    kama_dir[10:] = np.where(kama[10:] > kama[9:-1], 1, np.where(kama[10:] < kama[9:-1], -1, 0))
    
    # === RSI Calculation (momentum) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values (set to 50 as neutral)
    rsi[:14] = 50
    
    # === Choppiness Index Calculation (regime filter) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first period
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = hh - ll
    # Avoid division by zero
    chop = np.where(denominator > 0, 
                    100 * np.log10(sum_tr / denominator) / np.log10(14), 
                    50)  # neutral when no range
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for KAMA, 14 for RSI/CHOP)
    start_idx = max(30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trending market condition (CHOP < 38.2)
        trending = chop[i] < 38.2
        
        # Long condition: KAMA rising, RSI > 50, trending market
        if kama_dir[i] == 1 and rsi[i] > 50 and trending:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short condition: KAMA falling, RSI < 50, trending market
        elif kama_dir[i] == -1 and rsi[i] < 50 and trending:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: choppy market or opposite signal
        else:
            # Exit if choppy or signal reverses
            if position == 1 and (chop[i] >= 38.2 or kama_dir[i] == -1):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (chop[i] >= 38.2 or kama_dir[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0