#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: 1d KAMA trend direction + RSI extremes + Choppiness regime filter.
- KAMA adapts to market noise, reducing whipsaws in choppy/ranging markets
- RSI < 30 for long, RSI > 70 for short to capture mean reversion within trend
- Choppiness Index > 61.8 confirms ranging regime where mean reversion works best
- Designed for low trade frequency (target: 30-100 trades over 4 years) with edge in both bull and bear markets
- Uses 1w HTF for regime context to avoid counter-trend trades in strong trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for regime context
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on 1d (ER=10, fast=2, slow=30)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility > 0, change / np.sum(volatility, axis=0, keepdims=True) if volatility.ndim > 1 else change / (np.sum(volatility) + 1e-10), 0)
    # Simplified ER calculation for 1D array
    volatility_sum = np.nansum(volatility)
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    # Correct ER calculation: |current - close[n-period]| / sum |diff| over period
    period_er = 10
    change_er = np.zeros_like(close_1d)
    volatility_er = np.zeros_like(close_1d)
    for i in range(period_er, len(close_1d)):
        change_er[i] = np.abs(close_1d[i] - close_1d[i - period_er])
        volatility_er[i] = np.sum(np.abs(np.diff(close_1d[i - period_er:i + 1])))
    er = np.where(volatility_er > 0, change_er / volatility_er, 0)
    er[:period_er] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    rsi_period = 14
    avg_gain = wilders_smoothing(gain, rsi_period)
    avg_loss = wilders_smoothing(loss, rsi_period)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1w[0] - low_1w[0]
    
    # ATR(14) sum
    atr_period = 14
    atr_sum = np.zeros_like(tr)
    for i in range(atr_period, len(tr)):
        atr_sum[i] = np.sum(tr[i - atr_period + 1:i + 1])
    
    # Highest high and lowest low over period
    hh = np.zeros_like(high_1w)
    ll = np.zeros_like(low_1w)
    for i in range(atr_period, len(high_1w)):
        hh[i] = np.max(high_1w[i - atr_period + 1:i + 1])
        ll[i] = np.min(low_1w[i - atr_period + 1:i + 1])
    
    # Chop = 100 * log10(ATR_sum / (HH - LL)) / log10(period)
    chop = np.zeros_like(close_1w)
    for i in range(atr_period, len(close_1w)):
        if hh[i] > ll[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(atr_period)
        else:
            chop[i] = 50  # neutral when no range
    chop[:atr_period] = 50
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need ER period + KAMA, RSI, Chop periods)
    start_idx = max(period_er, rsi_period, atr_period) + 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA trend direction
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Choppiness regime: > 61.8 = ranging (good for mean reversion)
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price above KAMA (uptrend) AND RSI oversold AND ranging regime
            if price_above_kama and rsi_oversold and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) AND RSI overbought AND ranging regime
            elif price_below_kama and rsi_overbought and chop_ranging:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below KAMA OR RSI > 50 (mean reversion complete) OR chop < 50 (trending)
            if price_below_kama or rsi_aligned[i] > 50 or chop_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above KAMA OR RSI < 50 (mean reversion complete) OR chop < 50 (trending)
            if price_above_kama or rsi_aligned[i] < 50 or chop_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0