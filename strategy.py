#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, KAMA trend direction combined with RSI extremes and choppiness regime filter captures sustained moves while avoiding whipsaws in ranging markets. KAMA adapts to market efficiency, RSI<30/>70 identifies exhaustion points in trend, and Choppiness Index >61.8 confirms ranging conditions for mean-reversion exits. Works in bull via trend continuation and in bear via mean-reversion in ranges. Discrete sizing (0.25) limits fee drag. Target: 50-100 trades over 4 years.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # Efficiency Ratio: |net change| / sum(|changes|) over ER_period
    er_period = 10
    change = np.abs(np.diff(close, prepend=close[0]))
    net_change = np.abs(np.subtract(close, np.roll(close, er_period)))
    net_change[:er_period] = np.nan
    
    # Sum of absolute changes
    abs_changes = pd.Series(change).rolling(window=er_period, min_periods=er_period).sum().values
    er = np.where(abs_changes > 0, net_change / abs_changes, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]  # seed
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) ===
    chop_period = 14
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr / (hh - ll)) / np.log10(chop_period), 50)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup period
    start_idx = max(er_period, rsi_period, chop_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price above KAMA (uptrend) AND RSI < 30 (oversold) AND chop > 61.8 (ranging -> mean reversion long)
        long_condition = (close[i] > kama[i]) and (rsi[i] < 30) and (chop[i] > 61.8)
        # Short logic: price below KAMA (downtrend) AND RSI > 70 (overbought) AND chop > 61.8 (ranging -> mean reversion short)
        short_condition = (close[i] < kama[i]) and (rsi[i] > 70) and (chop[i] > 61.8)
        
        # Exit logic: trend reversal (price crosses KAMA) OR chop < 38.2 (strong trend -> follow trend with KAMA)
        exit_long = (close[i] < kama[i]) or (chop[i] < 38.2)
        exit_short = (close[i] > kama[i]) or (chop[i] < 38.2)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0