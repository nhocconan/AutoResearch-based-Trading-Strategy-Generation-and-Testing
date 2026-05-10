#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Overbought_Oversold_v2
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) from 1d timeframe to determine trend direction.
# Enters long when price crosses above KAMA in an uptrend with RSI < 30 (oversold) and volume confirmation (>1.5x 20-period average).
# Enters short when price crosses below KAMA in a downtrend with RSI > 70 (overbought) and volume confirmation.
# Exits when price crosses back across KAMA (trend reversal).
# Designed for low trade frequency to avoid fee drag, with trend-following bias and mean-reversion entries.

name = "4h_KAMA_Direction_RSI_Overbought_Oversold_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for KAMA trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate KAMA (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    vol = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])))
    er = np.where(vol != 0, change / vol, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) for overbought/oversold conditions
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA, RSI < 30 (oversold), volume confirmation
            if (close[i] > kama_aligned[i] and 
                close[i-1] <= kama_aligned[i-1] and 
                rsi[i] < 30 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA, RSI > 70 (overbought), volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  close[i-1] >= kama_aligned[i-1] and 
                  rsi[i] > 70 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA (trend reversal)
            if close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA (trend reversal)
            if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals