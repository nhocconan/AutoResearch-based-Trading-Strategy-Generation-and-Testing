#!/usr/bin/env python3
"""
12h_KAMA_Direction_1dRSI_Extremes_With_Volume
Hypothesis: Use KAMA on 12h for trend direction, combined with RSI extremes on 1d and volume spike. 
KAMA adapts to market noise, reducing whipsaw in choppy markets. RSI extremes on daily timeframe 
provide overextension signals for mean reversion, while volume confirms momentum. 
Designed for 12h timeframe to target 50-150 trades over 4 years, avoiding excessive trading.
Works in both bull and bear markets by adapting trend strength and using mean reversion in ranges.
"""

name = "12h_KAMA_Direction_1dRSI_Extremes_With_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 12h KAMA for trend direction
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily RSI(14)
    rsi_period = 14
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full_like(df_1d['close'].values, np.nan, dtype=np.float64)
    avg_loss = np.full_like(df_1d['close'].values, np.nan, dtype=np.float64)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    # Wilder smoothing
    for i in range(rsi_period + 1, len(df_1d)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get 12h volume
    volume = prices['volume'].values
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10 periods), RSI (14 days), volume EMA (20)
    start_idx = max(10, 20)  # KAMA needs 10, volume EMA needs 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) AND RSI oversold (<30) AND volume spike
            if close[i] > kama[i] and rsi_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) AND RSI overbought (>70) AND volume spike
            elif close[i] < kama[i] and rsi_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA OR RSI reaches overbought (>70)
            if close[i] < kama[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA OR RSI reaches oversold (<30)
            if close[i] > kama[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals