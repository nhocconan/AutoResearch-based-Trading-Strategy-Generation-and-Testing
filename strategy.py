#!/usr/bin/env python3
# 4h_1d_KAMA_Trend_RSI_MeanReversion
# Hypothesis: Use 1d KAMA for trend direction, 4h RSI for mean reversion entries, and volume confirmation.
# Long when: 1d KAMA rising (uptrend), 4h RSI < 30 (oversold), volume > 1.5x average.
# Short when: 1d KAMA falling (downtrend), 4h RSI > 70 (overbought), volume > 1.5x average.
# Exit when RSI returns to neutral (40-60) or trend changes.
# Designed for 4h timeframe with ~20-50 trades/year to avoid fee drag. Works in both bull and bear markets by following trend and buying dips/selling rallies.

name = "4h_1d_KAMA_Trend_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=er_length, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

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
    
    # Calculate 1d KAMA for trend
    close_1d = df_1d['close'].values
    kama_1d = calculate_kama(close_1d, er_length=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h RSI for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend (price above KAMA), oversold RSI, volume confirmation
            if (close[i] > kama_1d_aligned[i] and 
                rsi[i] < 30 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price below KAMA), overbought RSI, volume confirmation
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi[i] > 70 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or trend changes
            if rsi[i] > 40 or close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or trend changes
            if rsi[i] < 60 or close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals