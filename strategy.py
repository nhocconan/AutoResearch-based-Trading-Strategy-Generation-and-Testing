#!/usr/bin/env python3
"""
4h_1d_kama_rsi_volatility_breakout_v2
Hypothesis: KAMA trend direction combined with RSI momentum and volume confirmation on 4h.
Uses 1-day KAMA for trend filter, 4-hour RSI for entry timing, and volume spike for confirmation.
Designed to work in both bull and bear markets by filtering trades with trend alignment.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

name = "4h_1d_kama_rsi_volatility_breakout_v2"
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
    
    # Get daily data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # Parameters: ER length=10, Fast=2, Slow=30
    change = np.abs(np.subtract(close_1d[10:], close_1d[:-10]))
    volatility = np.sum(np.abs(np.diff(close_1d.reshape(-1, 10), axis=1)), axis=1)
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align lengths
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)
    
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI on 4h close (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > KAMA (uptrend) + RSI > 55 + volume spike
        if (close[i] > kama_aligned[i] and rsi[i] > 55 and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < KAMA (downtrend) + RSI < 45 + volume spike
        elif (close[i] < kama_aligned[i] and rsi[i] < 45 and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or RSI mean reversion
        elif position == 1 and (close[i] < kama_aligned[i] or rsi[i] < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_aligned[i] or rsi[i] > 50):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals