#!/usr/bin/env python3
# 12h_1d_KAMA_RSI_TrendFilter
# Hypothesis: KAMA trend direction on 12h + RSI(14) mean-reversion on 12h + volume spike confirmation.
# KAMA adapts to market noise, reducing whipsaws in choppy markets. RSI identifies overbought/oversold
# conditions within the trend. Volume spike confirms conviction. Works in bull/bear by following
# adaptive trend while avoiding extreme reversals. Target: 15-30 trades/year per symbol.

name = "12h_1d_KAMA_RSI_TrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast_len=2, slow_len=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.subtract.accumulate(change))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d KAMA for trend
    kama_1d = calculate_kama(close_1d, er_len=10, fast_len=2, slow_len=30)
    
    # Calculate 1d RSI for mean reversion
    rsi_1d = calculate_rsi(close_1d, period=14)
    
    # Calculate 1d volume average for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.8 * 1d average volume
        volume_spike = volume[i] > 1.8 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: above KAMA (uptrend), RSI oversold (<30), volume spike
            if close[i] > kama_1d_aligned[i] and rsi_1d[i] < 30 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: below KAMA (downtrend), RSI overbought (>70), volume spike
            elif close[i] < kama_1d_aligned[i] and rsi_1d[i] > 70 and volume_spike:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if below KAMA (trend change) or RSI overbought (>70)
            if close[i] < kama_1d_aligned[i] or rsi_1d[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if above KAMA (trend change) or RSI oversold (<30)
            if close[i] > kama_1d_aligned[i] or rsi_1d[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals