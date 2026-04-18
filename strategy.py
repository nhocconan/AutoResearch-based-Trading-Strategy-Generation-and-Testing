#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA direction + RSI(14) + Chop(14) regime filter.
- Long: KAMA rising, RSI(14) < 30 (oversold), Chop(14) > 61.8 (ranging market)
- Short: KAMA falling, RSI(14) > 70 (overbought), Chop(14) > 61.8 (ranging market)
- Exit: Opposite RSI condition (RSI > 50 for longs, RSI < 50 for shorts) or Chop < 38.2 (trending)
- Uses 1d Chop for regime filter to avoid whipsaws in strong trends.
- Designed for mean-reversion in ranging markets with controlled trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    if n < er_period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    change = np.abs(close[er_period:] - close[:-er_period])
    volatility = np.sum(np.abs(np.diff(close[:er_period+1])))
    er = np.zeros(n)
    er[er_period:] = change / volatility
    er[er == 0] = 0  # avoid division by zero
    
    # Smoothing Constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period):
    """Calculate Relative Strength Index."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.full(n, np.nan)
    rs[period:] = avg_gain[period:] / np.where(avg_loss[period:] == 0, 1e-10, avg_loss[period:])
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_chop(high, low, close, period):
    """Calculate Choppiness Index."""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr[period-1] = np.nanmean(tr[1:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    # Highest high and lowest low over period
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    for i in range(period-1, n):
        hh[i] = np.max(high[i-period+1:i+1])
        ll[i] = np.min(low[i-period+1:i+1])
    
    # Chop calculation
    chop = np.full(n, np.nan)
    for i in range(period-1, n):
        if hh[i] - ll[i] != 0:
            chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / np.log10(2) / (hh[i] - ll[i]))
        else:
            chop[i] = 50  # neutral when no range
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Chop regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chop (14-period) on 1d
    chop_14_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align Chop to 4h timeframe
    chop_14_1d_4h = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    # Calculate KAMA (10,2,30) on 4h
    kama = calculate_kama(close, 10, 2, 30)
    
    # Calculate RSI (14-period) on 4h
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need KAMA, RSI, and Chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_14_1d_4h[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Chop regime: > 61.8 = ranging (good for mean reversion), < 38.2 = trending
        chop_ranging = chop_14_1d_4h[i] > 61.8
        chop_trending = chop_14_1d_4h[i] < 38.2
        
        if position == 0:
            # Long: KAMA rising, RSI oversold, Chop ranging
            if kama_rising and rsi[i] < 30 and chop_ranging:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought, Chop ranging
            elif kama_falling and rsi[i] > 70 and chop_ranging:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 or Chop trending
            if rsi[i] > 50 or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 or Chop trending
            if rsi[i] < 50 or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Chop"
timeframe = "4h"
leverage = 1.0