#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter Strategy
KAMA identifies trend direction, RSI provides overbought/oversold signals,
Choppiness Index filters for trending vs ranging markets.
Works in both bull and bear markets by adapting to regime.
Target: 15-25 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_choppiness(high, low, close, window=14):
    """Choppiness Index: measures whether market is choppy (ranging) or trending"""
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = np.where(tr1 > 0, tr, 0)
    atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
    hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
    ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window), 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index
    chop = calculate_choppiness(high, low, close, window=14)
    
    # Get weekly close for trend filter
    close_1w = df_1w['close'].values
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA(10), RSI(14), Chop(14)
    start_idx = max(30, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        kama_now = kama[i]
        rsi_now = rsi[i]
        chop_now = chop[i]
        trend_1w = close_1w_aligned[i]
        
        # Chop filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_now < 61.8
        
        if position == 0:
            # Long: price > KAMA, RSI < 70 (not overbought), trending regime, weekly uptrend
            if price_now > kama_now and rsi_now < 70 and trending_regime and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price < KAMA, RSI > 30 (not oversold), trending regime, weekly downtrend
            elif price_now < kama_now and rsi_now > 30 and trending_regime and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: price < KAMA OR RSI > 70 (overbought) OR choppy market
            if price_now < kama_now or rsi_now > 70 or chop_now >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price > KAMA OR RSI < 30 (oversold) OR choppy market
            if price_now > kama_now or rsi_now < 30 or chop_now >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0