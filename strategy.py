#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Range_200MA_v1
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI for momentum confirmation, and 200-day MA for long-term bias. This combination aims to capture trending moves while avoiding false signals in ranging markets. Designed for low trade frequency (20-50 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA for trend direction
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        for i in range(len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        for i in range(1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Get 1d data for 200-day MA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-day MA
    ma200_1d = np.full_like(close_1d, np.nan)
    for i in range(199, len(close_1d)):
        ma200_1d[i] = np.mean(close_1d[i-199:i+1])
    
    # Calculate indicators on 4h data
    kama_val = kama(close, length=10, fast=2, slow=30)
    rsi_val = rsi(close, length=14)
    
    # Align 1d MA200 to 4h
    ma200_1d_aligned = align_htf_to_ltf(prices, df_1d, ma200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = 200  # MA200 needs 200 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(ma200_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val_i = kama_val[i]
        rsi_val_i = rsi_val[i]
        ma200 = ma200_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price above KAMA, RSI > 50, price above 200-day MA
            if close_val > kama_val_i and rsi_val_i > 50 and close_val > ma200:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short conditions: price below KAMA, RSI < 50, price below 200-day MA
            elif close_val < kama_val_i and rsi_val_i < 50 and close_val < ma200:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI < 40
            if close_val < kama_val_i or rsi_val_i < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI > 60
            if close_val > kama_val_i or rsi_val_i > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Direction_RSI_Range_200MA_v1"
timeframe = "4h"
leverage = 1.0