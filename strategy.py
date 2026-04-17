#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Confirm_v1
4-hour strategy using KAMA (Kaufman Adaptive Moving Average) as trend filter with RSI confirmation and volume spike.
Long when price > KAMA, RSI < 50, and volume spike (momentum in uptrend).
Short when price < KAMA, RSI > 50, and volume spike (momentum in downtrend).
Uses KAMA's adaptive nature to reduce whipsaw in choppy markets.
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
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    change = np.abs(np.diff(close, k=10))  # |close - close 10 periods ago|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute changes
    er = np.where(volatility != 0, change / volatility, 0)  # efficiency ratio
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)  # smoothing constant
    kama = np.full_like(close, np.nan)
    kama[kama_period] = close[kama_period]  # seed
    for i in range(kama_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14-period) ===
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    for i in range(rsi_period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Spike Detection ===
    vol_ma_period = 20
    vol_ma = np.full_like(volume, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    vol_ratio = volume / vol_ma
    vol_spike = vol_ratio > 1.5  # volume 50% above average
    
    signals = np.zeros(n)
    warmup = max(kama_period, rsi_period, vol_ma_period) + 5
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above KAMA, RSI < 50 (not overbought), volume spike
            if close[i] > kama[i] and rsi[i] < 50 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below KAMA, RSI > 50 (not oversold), volume spike
            elif close[i] < kama[i] and rsi[i] > 50 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or loss of momentum
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_RSI_Confirm_v1"
timeframe = "4h"
leverage = 1.0