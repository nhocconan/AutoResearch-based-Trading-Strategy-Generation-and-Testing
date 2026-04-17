#!/usr/bin/env python3
"""
4h_KAMA_Trend_Reversal_v1
KAMA trend direction on 4h + mean reversion entry at RSI extremes with volume confirmation.
Uses daily close trend filter (KAMA direction) for trend alignment,
RSI(14) for mean reversion entries, and volume spike for confirmation.
Designed to work in both bull and bear markets by combining trend following
with mean reversion at extreme RSI levels.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d KAMA (Kaufman Adaptive Moving Average) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 10:
            er[i] = np.sum(change[i-9:i+1]) / np.sum(volatility[i-9:i+1]) if np.sum(volatility[i-9:i+1]) > 0 else 0
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 4h RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === Align 1d KAMA to 4h timeframe ===
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above KAMA (uptrend) AND RSI oversold (<30) AND volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi[i] < 30 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below KAMA (downtrend) AND RSI overbought (>70) AND volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] > 70 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI overbought (>70) OR price crosses below KAMA
            if (rsi[i] > 70 or 
                close[i] < kama_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold (<30) OR price crosses above KAMA
            if (rsi[i] < 30 or 
                close[i] > kama_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Reversal_v1"
timeframe = "4h"
leverage = 1.0