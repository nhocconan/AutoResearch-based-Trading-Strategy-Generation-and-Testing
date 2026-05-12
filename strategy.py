#!/usr/bin/env python3
# 6H_KAMA_TREND_WITH_VOLUME_CONFIRMATION
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing a dynamic trend filter.
# Combined with volume confirmation on 6h timeframe, this strategy aims to capture sustained moves while avoiding chop.
# KAMA trend direction + volume spike triggers entries, with exits on trend reversal or volume drop.
# Works in both bull and bear markets by following the adaptive trend.
# Target: 15-35 trades/year on 6h timeframe to stay within fee-efficient range.

name = "6H_KAMA_TREND_WITH_VOLUME_CONFIRMATION"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on close prices
    # KAMA parameters: ER period=10, fast=2, slow=30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will be calculated properly below
    
    # Proper volatility calculation (sum of absolute changes over er_period)
    volatility = np.zeros_like(close)
    for i in range(er_period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i])))
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility != 0
    er[er_period:][mask[er_period:]] = change[mask[er_period:]] / volatility[er_period:][mask[er_period:]]
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume spike > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if KAMA not ready (need at least er_period + 1 values)
        if i < er_period + 1 or np.isnan(kama[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA with volume spike
            if close[i] > kama[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA with volume spike
            elif close[i] < kama[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or volume drops
            if close[i] < kama[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or volume drops
            if close[i] > kama[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals