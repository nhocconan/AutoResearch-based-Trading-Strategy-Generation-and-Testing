#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_And_Chop_Filter
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 4h for trend direction,
# enters in trend direction on pullbacks with volume confirmation (>1.5x 20-period average),
# and uses Choppiness Index (14-period) to avoid ranging markets (CHOP > 61.8 = range, avoid).
# Designed for low trade frequency (<150 total 4h trades) to minimize fee drag.
# Works in bull/bear markets by following adaptive trend while filtering noise.

name = "4h_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # KAMA (ER=10, FAST=2, SLOW=30) on close
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # This is incorrect, need rolling sum
    # Recompute volatility properly
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Choppiness Index (14-period)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr2[0] = tr1[0]
    tr3 = np.abs(np.roll(low, 1) - close)
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_high_low = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where((highest_high - lowest_low) != 0,
                    100 * np.log10(sum_high_low / (highest_high - lowest_low)) / np.log10(14),
                    50)
    
    # Trend filter: avoid ranging markets (CHOP > 61.8 = range)
    chop_filter = chop < 61.8  # Trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if np.isnan(kama[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > KAMA + volume spike + trending market
            if close[i] > kama[i] and volume_spike[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA + volume spike + trending market
            elif close[i] < kama[i] and volume_spike[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA OR chop > 61.8 (ranging)
            if close[i] < kama[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA OR chop > 61.8 (ranging)
            if close[i] > kama[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals