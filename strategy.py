#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) as regime filter to avoid whipsaws.
Enter long when KAMA upward, RSI > 50, and CHOP < 38.2 (trending market).
Enter short when KAMA downward, RSI < 50, and CHOP < 38.2.
Exit when opposite condition occurs.
Uses discrete position sizing (0.25) to minimize churn. Designed for 30-100 total trades over 4 years.
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
    
    # --- KAMA calculation (trend direction) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first 9 values where we don't have 10-period lookback
    er = np.full_like(change, np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(change) & ~np.isnan(volatility) & (volatility != 0))[0]
    if len(valid_idx) > 0:
        er[valid_idx] = change[valid_idx] / volatility[valid_idx]
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # --- RSI(14) calculation ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # --- Choppiness Index(14) ---
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    # Chop = 100 * log10(atr_sum / range_hl) / log10(14)
    chop = np.full_like(close, np.nan, dtype=float)
    valid = (atr_sum > 0) & (range_hl > 0)
    chop[valid] = 100 * np.log10(atr_sum[valid] / range_hl[valid]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (max of KAMA seed, RSI, Chop)
    start_idx = max(10, 14, 14)  # KAMA needs 10, RSI 14, Chop 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            i >= len(kama) or i >= len(rsi) or i >= len(chop)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: KAMA up, RSI > 50, Chop < 38.2 (trending)
        if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 38.2:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: KAMA down, RSI < 50, Chop < 38.2 (trending)
        elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 38.2:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite condition
        elif position == 1 and (kama[i] <= kama[i-1] or rsi[i] <= 50 or chop[i] >= 38.2):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (kama[i] >= kama[i-1] or rsi[i] >= 50 or chop[i] >= 38.2):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0