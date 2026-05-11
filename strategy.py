#!/usr/bin/env python3
"""
4h_1d_KAMA_Direction_With_RSI_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. 
Combined with RSI for momentum confirmation and volume filter to avoid false breaks.
Long when: KAMA upward, RSI > 50, volume above average
Short when: KAMA downward, RSI < 50, volume above average
Exit when: RSI crosses back to 50 or KAMA flips direction
Designed for 4-8 trades per month per symbol (48-96 over 4 years) to minimize fee drag.
Works in both bull (trend following) and bear (mean reversion via RSI extremes) markets.
"""

name = "4h_1d_KAMA_Direction_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for KAMA trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h data
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # --- 1d KAMA (Adaptive Trend) ---
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum of |diff| over 10 periods
    # Handle array shapes properly
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    
    # Avoid division by zero
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # --- 1d RSI (14) ---
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # first average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # --- Volume Confirmation (4h) ---
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(20, len(volume_4h)):
        vol_ma_20[i] = np.mean(volume_4h[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60  # for KAMA and RSI stability
    
    for i in range(start_idx, n):
        # Skip if any values are NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine KAMA direction (slope)
        kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        # Volume filter
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Enter long: KAMA up, RSI > 50 (bullish momentum), volume confirmation
            if kama_rising and rsi_1d_aligned[i] > 50 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down, RSI < 50 (bearish momentum), volume confirmation
            elif kama_falling and rsi_1d_aligned[i] < 50 and vol_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI drops below 50 OR KAMA turns down
                if rsi_1d_aligned[i] < 50 or not kama_rising:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI rises above 50 OR KAMA turns up
                if rsi_1d_aligned[i] > 50 or not kama_falling:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals