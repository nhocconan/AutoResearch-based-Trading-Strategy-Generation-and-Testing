#!/usr/bin/env python3
"""
6h_WMA_Cross_1dATR_Volume_Regime
Hypothesis: A fast/slow Weighted Moving Average crossover on 6h, filtered by 1d ATR-based volatility regime and volume confirmation, captures momentum in both bull and bear markets.
- WMA(9) > WMA(21) = bullish momentum, < = bearish
- 1d ATR ratio (current/20-period avg) > 1.2 = high volatility regime (favor trend following)
- Volume > 1.5x 20-period average confirms participation
Target: 20-50 trades/year to minimize fee drag while maintaining edge.
"""

name = "6h_WMA_Cross_1dATR_Volume_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wma(arr, period):
    """Weighted Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=np.float64)
    weights = np.arange(1, period + 1, dtype=np.float64)
    weights_sum = weights.sum()
    wma_vals = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1:i + 1]
        wma_vals[i] = np.dot(window, weights) / weights_sum
    return wma_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # WMA crossover on 6h
    wma_fast = wma(close, 9)
    wma_slow = wma(close, 21)
    
    # 1d ATR for volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 20-period average ATR
    atr_ma = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(wma_fast[i]) or np.isnan(wma_slow[i]) or np.isnan(atr_ratio_aligned[i]):
            signals[i] = 0.0
            continue
        
        wma_bullish = wma_fast[i] > wma_slow[i]
        wma_bearish = wma_fast[i] < wma_slow[i]
        high_vol_regime = atr_ratio_aligned[i] > 1.2
        vol_confirm = volume_filter[i]
        
        if position == 0:
            # LONG: WMA bullish crossover in high volatility regime with volume
            if wma_bullish and high_vol_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: WMA bearish crossover in high volatility regime with volume
            elif wma_bearish and high_vol_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: WMA turns bearish OR volatility drops
            if not wma_bullish or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: WMA turns bullish OR volatility drops
            if not wma_bearish or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals