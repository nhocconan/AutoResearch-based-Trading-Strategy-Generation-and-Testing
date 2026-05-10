#!/usr/bin/env python3
# 4h_KAMA_Trend_With_RSI_and_Volume_Filter
# Hypothesis: KAMA adapts to market noise, providing a reliable trend filter in both bull and bear markets.
# Combined with RSI (40-60 for trend strength) and volume confirmation, it filters false signals.
# Trades only in direction of KAMA trend with volume filter to reduce whipsaws.
# Target: 20-50 trades/year on 4h timeframe.

name = "4h_KAMA_Trend_With_RSI_and_Volume_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # KAMA (Kaufman Adaptive Moving Average) - 10 period
    # ER = |net change| / sum(|abs change|)
    # SSC = [ER * (fast - slow) + slow]^2
    # KAMA = previous KAMA + SSC * (price - previous KAMA)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Avoid division by zero
    sum_abs_change = np.sum(change) if np.sum(change) > 0 else 1
    er = np.abs(np.diff(close, prepend=close[0])) / sum_abs_change
    
    # Handle first element
    er[0] = 0
    
    fast_sc = 2 / (2 + 1)  # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    ssc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    ssc[0] = 0  # Initialize
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + ssc[i] * (close[i] - kama[i-1])
    
    # RSI (14 period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Handle division by zero (when avg_loss is 0)
    rsi = np.where(avg_loss == 0, 100, rsi)
    # Handle first 13 values
    rsi[:13] = 50
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30), RSI (14), volume MA (20)
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price relative to KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # RSI filter: 40-60 for trend strength (avoid extremes)
        rsi_mid = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: above KAMA + RSI in middle range + volume
            if above_kama and rsi_mid and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: below KAMA + RSI in middle range + volume
            elif below_kama and rsi_mid and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if not above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if not below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals