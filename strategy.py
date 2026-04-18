#!/usr/bin/env python3
"""
4h_KAMA_RSI_Trend_Reversal
Hypothesis: Capture trend reversals by combining Kaufman Adaptive Moving Average (KAMA) direction with RSI extremes, filtered by 1-day trend via EMA(50). KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trend changes. RSI identifies overbought/oversold conditions for mean reentry within the trend. Position size 0.25 targets ~25 trades/year to minimize fee drag. Works in bull markets by catching pullbacks in uptrends and in bear markets by catching bounces in downtrends.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    kama_period = 10
    fast = 2
    slow = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=kama_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA values
    kama = np.full_like(close, np.nan)
    kama[kama_period] = close[kama_period]  # seed
    for i in range(kama_period + 1, n):
        kama[i] = kama[i-1] + sc[i - kama_period] * (close[i] - kama[i-1])
    
    # RSI calculation (14-period)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initial average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
    
    rsi = np.full_like(close, 50.0)  # default neutral
    rs = np.zeros_like(avg_loss)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[rsi_period+1:] = 100 - (100 / (1 + rs[rsi_period+1:]))
    
    # 1d EMA trend filter (50-period)
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align KAMA and RSI to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kama_period + 1, rsi_period + 1, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA AND RSI < 30 (oversold) AND price > 1d EMA (uptrend)
            if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and \
               rsi_aligned[i] < 30 and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA AND RSI > 70 (overbought) AND price < 1d EMA (downtrend)
            elif close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and \
                 rsi_aligned[i] > 70 and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA OR RSI > 70 (overbought)
            if close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] or \
               rsi_aligned[i] > 70:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA OR RSI < 30 (oversold)
            if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] or \
               rsi_aligned[i] < 30:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Trend_Reversal"
timeframe = "4h"
leverage = 1.0