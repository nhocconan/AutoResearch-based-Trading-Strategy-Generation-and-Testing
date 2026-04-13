#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Trend_Filter
Hypothesis: Use KAMA trend direction on daily timeframe filtered by weekly RSI extremes
and volume confirmation to capture sustained moves in both bull and bear markets.
KAMA adapts to market noise, reducing false signals in chop. Weekly RSI avoids
overbought/oversold exhaustion. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.subtract.accumulate(change))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
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
    
    # Get weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.concatenate([ [np.nan], rsi_1w[1:] ])  # align with original index
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily KAMA for trend direction
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    kama_dir = np.where(close > kama, 1, -1)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(kama_dir[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: KAMA up, weekly RSI not overbought (<70), volume expansion
        long_condition = (kama_dir[i] == 1) and (rsi_1w_aligned[i] < 70) and volume_expansion[i]
        
        # Short: KAMA down, weekly RSI not oversold (>30), volume expansion
        short_condition = (kama_dir[i] == -1) and (rsi_1w_aligned[i] > 30) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_KAMA_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0