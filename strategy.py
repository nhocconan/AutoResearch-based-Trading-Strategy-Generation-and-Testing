#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime.
Long when KAMA trending up (price > KAMA) + RSI > 50 + Chop > 61.8 (range) for mean reversion to upside.
Short when KAMA trending down (price < KAMA) + RSI < 50 + Chop > 61.8 (range) for mean reversion to downside.
Exit when price crosses KAMA or Chop < 38.2 (trend) to avoid trend following in chop.
Designed for low frequency (7-25 trades/year) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.concatenate([[0], volatility[1:]])  # align length
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, window=14):
    """Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window, min_periods=window).mean().values
    
    max_high = pd.Series(high).rolling(window, min_periods=window).max().values
    min_low = pd.Series(low).rolling(window, min_periods=window).min().values
    
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(atr[i] * window / (max_high[i] - min_low[i])) / np.log10(window)
        else:
            chop[i] = 50
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for chop (use weekly chop for regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Chop for regime filter
    chop_1w = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate daily KAMA for trend
    kama = calculate_kama(close)
    
    # Calculate daily RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA (30), RSI (14), Chop (14)
    start_idx = max(30, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        kama_now = kama[i]
        rsi_now = rsi[i]
        chop_now = chop_1w_aligned[i]
        
        # Regime filter: Chop > 61.8 = range (good for mean reversion)
        is_range = chop_now > 61.8
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI > 50 + range regime
            if price_now > kama_now and rsi_now > 50 and is_range:
                signals[i] = size
                position = 1
            # Short: price < KAMA (downtrend) + RSI < 50 + range regime
            elif price_now < kama_now and rsi_now < 50 and is_range:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA OR chop < 38.2 (trend regime)
            if price_now < kama_now or chop_now < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA OR chop < 38.2 (trend regime)
            if price_now > kama_now or chop_now < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_ChopRegime"
timeframe = "1d"
leverage = 1.0