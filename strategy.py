#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Filter
Strategy: Trade with KAMA trend direction on 4h, confirmed by volume spike and RSI filter.
Long: KAMA trending up + RSI > 50 + volume > 1.5x average
Short: KAMA trending down + RSI < 50 + volume > 1.5x average
Exit: Trend reversal
Position size: 0.25
Designed to capture trending moves with volume confirmation to avoid whipsaws.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.abs(change)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate KAMA and RSI
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    rsi = calculate_rsi(close, length=14)
    
    # Volume confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for KAMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # KAMA trend: up if current close > KAMA, down if close < KAMA
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI filter: >50 for long, <50 for short
        rsi_long = rsi[i] > 50
        rsi_short = rsi[i] < 50
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + volume spike
            if kama_up and rsi_long and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + volume spike
            elif kama_down and rsi_short and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down or RSI < 50
            if not kama_up or not rsi_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up or RSI > 50
            if not kama_down or not rsi_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0