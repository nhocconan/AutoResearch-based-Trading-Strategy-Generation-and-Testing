#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction with 1w RSI filter and volume confirmation
# KAMA adapts to market efficiency, reducing whipsaws in choppy markets.
# 1w RSI > 50 for long, < 50 for short ensures alignment with weekly momentum.
# Volume confirmation ensures institutional participation.
# Targets 10-25 trades per year (~40-100 total over 4 years) to minimize fee drift.

name = "1d_KAMA_1wRSI_Volume"
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
    
    # Get 1w data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate KAMA on 1d
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if close > kama, -1 if close < kama
    kama_dir = np.where(close > kama, 1, np.where(close < kama, -1, 0))
    
    # Calculate RSI on 1w
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain and loss over 14 periods
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # RSI filter: >50 for long, <50 for short
    rsi_long = rsi_1w > 50
    rsi_short = rsi_1w < 50
    
    # Align to 1d
    kama_dir_aligned = align_htf_to_ltf(prices, df_1w, kama_dir.astype(float))
    rsi_long_aligned = align_htf_to_ltf(prices, df_1w, rsi_long.astype(float))
    rsi_short_aligned = align_htf_to_ltf(prices, df_1w, rsi_short.astype(float))
    
    # Volume confirmation on 1d: volume > 1.5 * 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure sufficient data for volume MA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_long_aligned[i]) or 
            np.isnan(rsi_short_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA up, RSI > 50, volume confirmation
            if kama_dir_aligned[i] == 1 and rsi_long_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down, RSI < 50, volume confirmation
            elif kama_dir_aligned[i] == -1 and rsi_short_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down or RSI < 50
            if kama_dir_aligned[i] == -1 or not rsi_long_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up or RSI > 50
            if kama_dir_aligned[i] == 1 or not rsi_short_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals