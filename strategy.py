#!/usr/bin/env python3
"""
4h_KAMA_Direction_Plus_RSI_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction and RSI for overbought/oversold entries in the direction of the trend. Designed for low trade frequency (~25/year) to minimize fee drag on 4h timeframe. Works in bull/bear by only taking long entries in uptrends and short entries in downtrends, avoiding counter-trend trades.
"""

name = "4h_KAMA_Direction_Plus_RSI_Filter"
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
    volume = prices['volume'].values
    
    # === KAMA Trend Direction (10-period ER, 2/30 smoothing) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Spike Filter (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers KAMA and RSI calculation)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend) and RSI < 30 (oversold) with volume spike
            if (close[i] > kama[i] and 
                rsi[i] < 30 and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Price below KAMA (downtrend) and RSI > 70 (overbought) with volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price crosses back through KAMA
            if position == 1:
                if close[i] < kama[i]:  # Exit long if price crosses below KAMA
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > kama[i]:  # Exit short if price crosses above KAMA
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals