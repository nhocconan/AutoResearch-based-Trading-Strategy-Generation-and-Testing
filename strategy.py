#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_RSI_Momentum
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) as adaptive trend filter on 12h chart, combined with RSI momentum to catch trend continuations in both bull and bear markets. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing sustained moves.
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
    
    # Get 1d data for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d for trend filter
    close_1d = df_1d['close'].values
    # KAMA parameters: ER period=10, fast=2, slow=30
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Align KAMA to 12h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI on 1d for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi
    
    # Align RSI to 12h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for KAMA, RSI, volume MA
    start_idx = max(30, 20)  # KAMA needs 30, RSI needs ~14, VolMA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol_conf = vol_confirm[i]
        
        if position == 0:
            # Long: price above KAMA and RSI > 50 with volume confirmation
            if close[i] > kama_val and rsi_val > 50 and vol_conf:
                signals[i] = size
                position = 1
            # Short: price below KAMA and RSI < 50 with volume confirmation
            elif close[i] < kama_val and rsi_val < 50 and vol_conf:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA or RSI < 40
            if close[i] < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above KAMA or RSI > 60
            if close[i] > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_Trend_With_RSI_Momentum"
timeframe = "12h"
leverage = 1.0