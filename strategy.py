#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Plus_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing a reliable trend direction.
# Combined with RSI for momentum and Choppiness Index to avoid ranging markets.
# Works in both bull and bear markets by following the adaptive trend only when
# momentum confirms and the market is trending (not choppy).
# Expects low trade frequency due to triple confluence requirement.

name = "4h_KAMA_Direction_RSI_Plus_Chop_Filter"
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
    
    # === KAMA (10, 2, 30) for adaptive trend direction ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility sum
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # === RSI (14) for momentum confirmation ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) for regime filter ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # ATR (14)
    atr = np.full_like(close, np.nan)
    atr[13] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # Sum of ATR over 14 periods
    sum_atr = np.full_like(close, np.nan)
    sum_atr[13] = np.sum(atr[1:15])
    for i in range(15, n):
        sum_atr[i] = sum_atr[i-1] - atr[i-14] + atr[i]
    # Highest high and lowest low over 14 periods
    hh = np.full_like(close, np.nan)
    ll = np.full_like(close, np.nan)
    for i in range(14, n):
        hh[i] = np.max(high[i-13:i+1])
        ll[i] = np.min(low[i-13:i+1])
    # Chop calculation
    chop = np.full_like(close, np.nan)
    for i in range(14, n):
        if sum_atr[i] > 0 and hh[i] != ll[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (hh[i] - ll[i])) / np.log10(14)
    
    # === Chop Thresholds ===
    chop_below = chop < 38.2  # Trending
    chop_above = chop > 61.8  # Ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA (uptrend) + RSI > 50 (bullish momentum) + Trending market (chop < 38.2)
            if close[i] > kama[i] and rsi[i] > 50 and chop_below[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + RSI < 50 (bearish momentum) + Trending market (chop < 38.2)
            elif close[i] < kama[i] and rsi[i] < 50 and chop_below[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below KAMA (trend change) or RSI < 40 (momentum loss) or ranging market
            if close[i] < kama[i] or rsi[i] < 40 or chop_above[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA (trend change) or RSI > 60 (momentum loss) or ranging market
            if close[i] > kama[i] or rsi[i] > 60 or chop_above[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals