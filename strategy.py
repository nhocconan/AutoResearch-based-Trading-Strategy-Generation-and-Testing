#!/usr/bin/env python3
# Hypothesis: 1d KAMA (Kaufman Adaptive Moving Average) direction + RSI(14) + weekly volatility regime filter.
# KAMA adapts to market noise - follows price closely in trending markets, stays flat in ranging markets.
# In trending markets (weekly ATR > 20-period average), we take KAMA direction signals.
# In ranging markets (weekly ATR <= 20-period average), we mean-revert at RSI extremes.
# This dual-regime approach works in both bull and bear markets by adapting to volatility conditions.
# Weekly volatility filter prevents whipsaws in low-volatility environments.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volatility regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly ATR for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1w = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly volatility regime to daily
    vol_regime = atr_1w > atr_ma_1w  # True = trending, False = ranging
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime.astype(float))
    
    # Daily KAMA
    kama = _kama(close, er_length=10, fast_sc=2, slow_sc=30)
    kama_dir = np.where(close > kama, 1, -1)  # 1 = above KAMA (bullish), -1 = below KAMA (bearish)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_direction = kama_dir[i]
        rsi_value = rsi[i]
        is_trending = vol_regime_aligned[i] > 0.5  # Weekly volatility regime
        
        if is_trending:
            # Trending regime: follow KAMA direction
            long_entry = kama_direction == 1 and position <= 0
            short_entry = kama_direction == -1 and position >= 0
            # Exit when KAMA direction changes
            long_exit = kama_direction == -1 and position == 1
            short_exit = kama_direction == 1 and position == -1
        else:
            # Ranging regime: mean reversion at RSI extremes
            long_entry = rsi_value < 30 and position <= 0  # Oversold
            short_entry = rsi_value > 70 and position >= 0  # Overbought
            # Exit when RSI returns to neutral zone
            long_exit = rsi_value > 50 and position == 1
            short_exit = rsi_value < 50 and position == -1
        
        # Handle entries and exits
        if long_entry:
            signals[i] = 0.25
            position = 1
        elif short_entry:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_VolatilityRegime"
timeframe = "1d"
leverage = 1.0