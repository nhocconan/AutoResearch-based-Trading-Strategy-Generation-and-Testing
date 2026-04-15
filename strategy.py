# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h KAMA + RSI with Chop Regime Filter
- Uses Kaufman Adaptive Moving Average (KAMA) on 6h for trend direction
- Uses RSI(14) on 6h for momentum confirmation
- Uses Choppiness Index (CHOP) on 1d to filter regimes: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
- In trending regime (CHOP < 38.2): follow KAMA direction
- In ranging regime (CHOP > 61.8): mean revert at RSI extremes (RSI < 30 long, RSI > 70 short)
- Works in bull markets (trend following) and bear markets (mean reversion in ranges)
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ===== KAMA calculation on 6h =====
    # Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(close - np.roll(close, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will be replaced with proper calculation
    
    # Proper volatility calculation (sum of absolute changes over 10 periods)
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for fast EMA = 2
    slow_sc = 2 / (30 + 1)  # for slow EMA = 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ===== RSI calculation on 6h =====
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # same length as close
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    # First average (simple average)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Subsequent averages (Wilder's smoothing)
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== 1d Choppiness Index (CHOP) for regime filter =====
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR (14-period)
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:15])  # First ATR
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Maximum and minimum close over 14 periods
    max_close_1d = np.zeros_like(close_1d)
    min_close_1d = np.zeros_like(close_1d)
    
    for i in range(13, len(close_1d)):
        max_close_1d[i] = np.max(close_1d[i-13:i+1])
        min_close_1d[i] = np.min(close_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if atr_1d[i] > 0 and (max_close_1d[i] - min_close_1d[i]) > 0:
            chop[i] = 100 * np.log10(np.sum(tr[i-13:i+1]) / atr_1d[i]) / np.log10(14)
        else:
            chop[i] = 50  # Neutral value
    
    # Align KAMA, RSI, and CHOP to 6h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # Using 1d index for alignment
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN or not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            continue
        
        chop_val = chop_aligned[i]
        
        if chop_val < 38.2:  # Trending regime - follow KAMA
            # Long when price above KAMA
            if close[i] > kama_aligned[i] and position <= 0:
                position = 1
                signals[i] = base_size
            # Short when price below KAMA
            elif close[i] < kama_aligned[i] and position >= 0:
                position = -1
                signals[i] = -base_size
                
        elif chop_val > 61.8:  # Ranging regime - mean revert at RSI extremes
            # Long when RSI oversold
            if rsi_aligned[i] < 30 and position <= 0:
                position = 1
                signals[i] = base_size
            # Short when RSI overbought
            elif rsi_aligned[i] > 70 and position >= 0:
                position = -1
                signals[i] = -base_size
                
        # Exit conditions: regime change or opposite signal
        if position == 1:
            # Exit long if: chop shifts to trending and price below KAMA, 
            # or chop shifts to ranging and RSI > 50, or RSI > 70 in ranging
            if ((chop_val < 38.2 and close[i] < kama_aligned[i]) or
                (chop_val > 61.8 and (rsi_aligned[i] > 50 or rsi_aligned[i] > 70))):
                position = 0
                signals[i] = 0.0
                
        elif position == -1:
            # Exit short if: chop shifts to trending and price above KAMA,
            # or chop shifts to ranging and RSI < 50, or RSI < 30 in ranging
            if ((chop_val < 38.2 and close[i] > kama_aligned[i]) or
                (chop_val > 61.8 and (rsi_aligned[i] < 50 or rsi_aligned[i] < 30))):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_KAMA_RSI_Chop_Regime"
timeframe = "6h"
leverage = 1.0