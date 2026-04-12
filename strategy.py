#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_Filter_v1
Hypothesis: Use 1d KAMA (Kaufman Adaptive Moving Average) to detect trend direction, filtered by 1w RSI for overbought/oversold conditions.
Long when KAMA turns upward and 1w RSI < 50 (avoiding overbought); short when KAMA turns downward and 1w RSI > 50 (avoiding oversold).
Exit when KAMA reverses direction. Uses volume confirmation (1.5x 20-bar average) to filter false signals.
Designed for low trade frequency (<10/year) by requiring trend alignment and volume confirmation.
Works in bull via KAMA uptrend and in bear via KAMA downtrend, with RSI filter preventing entries at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W RSI(14) FOR OVERBOUGHT/OVERSOLD FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # === 1D KAMA FOR TREND DIRECTION ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Volatility needs to be sum of absolute changes over 10 periods
    volatility_sum = np.zeros(n)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += np.abs(close[i] - (close[i-1] if i > 0 else close[i]))
        if i >= 10:
            vol_sum -= np.abs(close[i-10] - (close[i-11] if i >= 11 else 0))
        if i >= 9:
            volatility_sum[i] = vol_sum
    
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.zeros(n)
    kama_dir[0] = 0
    for i in range(1, n):
        if kama[i] > kama[i-1]:
            kama_dir[i] = 1
        elif kama[i] < kama[i-1]:
            kama_dir[i] = -1
        else:
            kama_dir[i] = kama_dir[i-1]
    
    # Align 1w RSI to 1d
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume average (20-period for confirmation)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if indicators not available
        if np.isnan(rsi_1w_aligned[i]) or vol_avg[i] == 0.0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Entry conditions
        long_setup = (kama_dir[i] == 1) and (rsi_1w_aligned[i] < 50) and vol_confirm
        short_setup = (kama_dir[i] == -1) and (rsi_1w_aligned[i] > 50) and vol_confirm
        
        # Exit when KAMA reverses direction
        exit_long = (kama_dir[i] == -1)
        exit_short = (kama_dir[i] == 1)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals