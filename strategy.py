#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Trend_Filter_v2
# Hypothesis: KAMA trend direction combined with RSI momentum and volume confirmation.
# Uses adaptive KAMA to filter noise, RSI for momentum strength, and volume spike for confirmation.
# Designed for low trade frequency (~25-35/year) to minimize fee decay while capturing trends in bull/bear markets.

name = "4h_KAMA_Direction_RSI_Trend_Filter_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_len=10, fast_len=2, slow_len=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
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
    volume = prices['volume'].values
    
    # 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA for trend direction (adaptive smoothing)
    kama_1d = kama(close_1d, er_len=10, fast_len=2, slow_len=30)
    kama_dir = np.where(kama_1d > np.roll(kama_1d, 1), 1, -1)  # 1=rising, -1=falling
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir.astype(float))
    
    # RSI for momentum (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50 (bullish momentum), volume spike
            if kama_dir_aligned[i] > 0 and rsi_aligned[i] > 50 and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50 (bearish momentum), volume spike
            elif kama_dir_aligned[i] < 0 and rsi_aligned[i] < 50 and volume[i] > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns down OR RSI < 40 (loss of momentum)
            if kama_dir_aligned[i] < 0 or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns up OR RSI > 60 (loss of bearish momentum)
            if kama_dir_aligned[i] > 0 or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals