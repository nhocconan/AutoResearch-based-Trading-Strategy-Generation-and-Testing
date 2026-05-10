#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: Kaufman's Adaptive Moving Average (KAMA) determines trend direction on daily timeframe.
# RSI(14) provides overbought/oversold signals for entry timing. Choppiness Index filters for trending markets only.
# Designed for 1d to achieve 7-25 trades/year, suitable for both bull and bear markets by avoiding ranging conditions.

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_fast=2, er_slow=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = np.abs(np.subtract(high, low))
    tr2 = np.abs(np.subtract(high, np.roll(close, 1)))
    tr3 = np.abs(np.subtract(low, np.roll(close, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    cpi = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(period)
    # Handle division by zero and edge cases
    cpi = np.where((highest_high - lowest_low) != 0, cpi, 50)
    return cpi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA for trend direction
    kama_1d = calculate_kama(close_1d)
    
    # RSI for overbought/oversold
    rsi_1d = calculate_rsi(close_1d)
    
    # Choppiness Index for regime filter
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d)
    
    # Align all indicators to lower timeframe (wait for 1d bar to close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI < 30 (oversold), trending market (CHOP < 61.8)
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 30 and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI > 70 (overbought), trending market (CHOP < 61.8)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 70 and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI > 70 (overbought)
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI < 30 (oversold)
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals