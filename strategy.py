#!/usr/bin/env python3
"""
4h_KAMA_RSI_Trend_With_Volume_Filter_v1
Hypothesis: KAMA trend direction combined with RSI momentum and volume confirmation captures sustained moves while avoiding whipsaws. Works in bull via trend-following and bear via mean-reversion at extremes. Designed for ~25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA trend filter (12h)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Calculate ER and SC for KAMA
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.abs(np.diff(close_12h))
    er = np.zeros_like(close_12h)
    for i in range(10, len(close_12h)):  # ER period=10
        if np.sum(volatility[i-9:i+1]) > 0:
            er[i] = change[i] / np.sum(volatility[i-9:i+1])
        else:
            er[i] = 0
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # RSI momentum (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(volume_spike[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_12h_aligned[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, volume spike
            if price > kama_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, volume spike
            elif price < kama_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below KAMA or RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above KAMA or RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_RSI_Trend_With_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0