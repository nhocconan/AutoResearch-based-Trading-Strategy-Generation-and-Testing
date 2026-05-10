#!/usr/bin/env python3
# 4h_KAMA_Trend_1d_RSI_Extreme
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 4h captures trend with low whipsaw,
# combined with 1d RSI extremes (>70 or <30) for mean-reversion entries in the direction of trend.
# Works in bull markets via buying dips in uptrend and in bear via selling rallies in downtrend.
# Low trade frequency expected due to dual condition (trend + extreme RSI).

name = "4h_KAMA_Trend_1d_RSI_Extreme"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_length=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def rsi(close, length=14):
    """Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for RSI (extreme filter)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RSI on daily timeframe
    rsi_1d = rsi(df_1d['close'].values, length=14)
    
    # Align RSI to 4h timeframe (no look-ahead)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 4h data for KAMA trend
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA on 4h for trend
    kama_4h = kama(close, er_length=10, fast=2, slow=30)
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14) + KAMA (30) + vol EMA (20)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(kama_4h[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) AND RSI < 30 (oversold) AND volume filter
            if close[i] > kama_4h[i] and rsi_1d_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) AND RSI > 70 (overbought) AND volume filter
            elif close[i] < kama_4h[i] and rsi_1d_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < KAMA (trend change) OR RSI > 70 (overbought)
            if close[i] < kama_4h[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA (trend change) OR RSI < 30 (oversold)
            if close[i] > kama_4h[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals