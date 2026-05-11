#!/usr/bin/env python3
"""
1d_KAMA_Trend_Regime_Filter
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 1d to capture trend direction,
combined with RSI(14) for momentum and Choppiness Index for regime filter.
In trending markets (CHOP < 38.2), follow KAMA direction with RSI confirmation.
In ranging markets (CHOP > 61.8), mean-revert at RSI extremes.
This adapts to both bull and bear regimes by using volatility-based trend detection.
"""

name = "1d_KAMA_Trend_Regime_Filter"
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
    
    # === KAMA Calculation (trend) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros(n)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    avg_loss[avg_loss == 0] = 1e-10
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) ===
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    
    # ATR(14)
    atr[13] = np.mean(tr[1:14])
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Highest high and lowest low over 14 periods
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(n):
        start = max(0, i-13)
        hh[i] = np.max(high[start:i+1])
        ll[i] = np.min(low[start:i+1])
    
    # Chop = 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    sum_atr14 = np.zeros(n)
    for i in range(13, n):
        sum_atr14[i] = np.sum(atr[i-13:i+1])
    
    chop = np.zeros(n)
    hh_ll = hh - ll
    hh_ll[hh_ll == 0] = 1e-10
    chop[13:] = 100 * np.log10(sum_atr14[13:] / hh_ll[13:]) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(hh[i]) or np.isnan(ll[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market: follow KAMA direction with RSI confirmation
            if chop[i] < 38.2:  # Trending
                if close[i] > kama[i] and rsi[i] > 50 and rsi[i] < 70:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and rsi[i] < 50 and rsi[i] > 30:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: mean reversion at RSI extremes
            elif chop[i] > 61.8:  # Ranging
                if rsi[i] < 30 and close[i] > ll[i]:  # Oversold and above low
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and close[i] < hh[i]:  # Overbought and below high
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: trend change or overbought in range
            if chop[i] < 38.2:  # Trending
                if close[i] < kama[i] or rsi[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging
                if rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: trend change or oversold in range
            if chop[i] < 38.2:  # Trending
                if close[i] > kama[i] or rsi[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging
                if rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals