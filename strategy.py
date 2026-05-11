#!/usr/bin/env python3
"""
12h_1d_KAMA_RSI_TrendFilter_Volume
Hypothesis: Uses KAMA trend direction from 12h combined with 1d RSI for momentum and 1d volume confirmation.
KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends.
RSI(14) on 1d filters for overbought/oversold conditions, and volume spike confirms institutional interest.
Designed for low trade frequency by requiring alignment of adaptive trend, momentum, and volume.
Works in bull markets (trend following) and bear markets (mean reversion via RSI extremes).
"""

name = "12h_1d_KAMA_RSI_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d data for RSI and volume ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # --- KAMA on 12h (adaptive trend) ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_12h = kama
    
    # --- 1d RSI(14) ---
    rsi_period = 14
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # --- 1d Volume Spike (20-period average) ---
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_1d
    vol_ratio_1d = np.nan_to_num(vol_ratio_1d, nan=1.0)
    vol_ratio_12h = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]) or 
            np.isnan(vol_ratio_12h[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: institutional participation
        volume_spike = vol_ratio_12h[i] > 1.8
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI not overbought + volume spike
            if (close[i] > kama_12h[i] and 
                rsi_12h[i] < 70 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI not oversold + volume spike
            elif (close[i] < kama_12h[i] and 
                  rsi_12h[i] > 30 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: loss of trend or extreme RSI
            if position == 1:
                if close[i] < kama_12h[i] or rsi_12h[i] > 80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] > kama_12h[i] or rsi_12h[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals