#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA adapts to market noise, providing a reliable trend filter. In trending markets (KAMA slope aligned with price), RSI extremes signal mean-reversion entries. Works in both bull and bear by trading pullbacks in the direction of the higher timeframe trend. Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=er_length, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
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
    rsi_input = pd.Series(close)
    delta = rsi_input.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    kama = calculate_kama(df_1d['close'].values)
    kama_slope = np.diff(kama, prepend=kama[0])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_slope)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        if np.isnan(kama_aligned[i]) or np.isnan(kama_slope_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above KAMA in uptrend, below KAMA in downtrend
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        kama_rising = kama_slope_aligned[i] > 0
        kama_falling = kama_slope_aligned[i] < 0
        
        # RSI conditions for mean-reversion entries
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Long: uptrend + RSI oversold
        if price_above_kama and kama_rising and rsi_oversold and position != 1:
            position = 1
            signals[i] = position_size
        # Short: downtrend + RSI overbought
        elif price_below_kama and kama_falling and rsi_overbought and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] >= 40:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] <= 60:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_KAMA_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0