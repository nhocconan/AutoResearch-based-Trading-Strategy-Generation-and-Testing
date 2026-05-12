#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Trend_Follow
# Hypothesis: Use 1d KAMA for trend direction and RSI for entry timing on 1d chart.
# Long when KAMA rising and RSI < 30, short when KAMA falling and RSI > 70.
# Designed to work in both bull and bear markets by following trend with mean-reversion entries.
# Target: 10-25 trades/year to minimize fee drag on higher timeframe.

name = "1d_KAMA_Direction_RSI_Trend_Follow"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    
    # Avoid division by zero
    volatility_sum = np.nansum(np.lib.stride_tricks.sliding_window_view(volatility, er_len), axis=1)
    change_sum = np.nansum(np.lib.stride_tricks.sliding_window_view(change, er_len), axis=1)
    
    er = np.divide(change_sum, volatility_sum, out=np.zeros_like(change_sum), where=volatility_sum!=0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.full_like(close, np.nan)
    kama[er_len-1] = close[er_len-1]
    
    for i in range(er_len, len(close)):
        if not np.isnan(sc[i-er_len+1]):
            kama[i] = kama[i-1] + sc[i-er_len+1] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1d KAMA for trend direction ===
    kama = calculate_kama(close, er_len=10, fast=2, slow=30)
    
    # === RSI for entry timing ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.nanmean(gain[1:14])
    avg_loss[13] = np.nanmean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from KAMA slope
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # LONG: KAMA rising and RSI oversold
            if kama_rising and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling and RSI overbought
            elif kama_falling and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI overbought
            if not kama_rising or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI oversold
            if not kama_falling or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals