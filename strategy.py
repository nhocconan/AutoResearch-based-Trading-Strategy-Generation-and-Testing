#!/usr/bin/env python3
# 1d_KAMA_Trend_Filter
# Hypothesis: 1-day trend following using Kaufman's Adaptive Moving Average (KAMA) with volume confirmation and ATR-based stoploss.
# KAMA adapts to market noise, reducing whipsaws in choppy markets and capturing trends in both bull and bear regimes.
# Volume filter ensures breakout strength. Designed for 1d to achieve 7-25 trades/year, suitable for both bull and bear markets.

name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA ( Kaufman's Adaptive Moving Average )
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else np.zeros_like(change)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # Calculate ATR for stoploss
    def atr(high, low, close, length=14):
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr_vals = np.zeros_like(close)
        for i in range(1, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (length-1) + tr[i]) / length
        return atr_vals
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly KAMA for trend filter (slower to avoid whipsaws)
    wkama = kama(close_1w, length=10, fast=2, slow=30)
    wkama_aligned = align_htf_to_ltf(prices, df_1w, wkama)
    
    # Daily KAMA for entry signal
    dkama = kama(close, length=10, fast=2, slow=30)
    
    # ATR for volatility filtering and stoploss
    atr_vals = atr(high, low, close, length=14)
    
    # Volume confirmation: 20-day average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(wkama_aligned[i]) or np.isnan(dkama[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_vals[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above daily KAMA, above weekly KAMA (trend filter), strong volume
            if close[i] > dkama[i] and close[i] > wkama_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below daily KAMA, below weekly KAMA (trend filter), strong volume
            elif close[i] < dkama[i] and close[i] < wkama_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below daily KAMA or ATR-based stoploss
            if close[i] < dkama[i] or close[i] < (prices['close'].values[i-1] - 2.0 * atr_vals[i-1] if i > 0 else close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above daily KAMA or ATR-based stoploss
            if close[i] > dkama[i] or close[i] > (prices['close'].values[i-1] + 2.0 * atr_vals[i-1] if i > 0 else close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals