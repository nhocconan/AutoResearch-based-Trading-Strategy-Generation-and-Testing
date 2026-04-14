#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day KAMA direction with RSI filter and volume confirmation
# - Long when KAMA turns upward, RSI > 50, and volume > 1.5x 24-period average
# - Short when KAMA turns downward, RSI < 50, and volume > 1.5x 24-period average
# - KAMA adapts to market noise, reducing whipsaws in ranging markets
# - Volume confirmation ensures breakouts have conviction
# - RSI filter avoids extremes and focuses on momentum continuation
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Works in both bull and bear markets by adapting to volatility regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=1)  # 10-period sum of absolute changes
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align with change
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    # Initial averages
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 24-period average (1 day of 12h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    # Align KAMA and RSI to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Determine KAMA direction (upward/downward)
        kama_dir = 1 if kama_aligned[i] > kama_aligned[i-1] else -1
        
        if position == 0:
            # Long: KAMA turning up, RSI > 50, volume confirmation
            if (kama_dir == 1 and 
                rsi_aligned[i] > 50 and
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: KAMA turning down, RSI < 50, volume confirmation
            elif (kama_dir == -1 and 
                  rsi_aligned[i] < 50 and
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: KAMA turns down or RSI < 40 (momentum fade)
            if kama_dir == -1 or rsi_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: KAMA turns up or RSI > 60 (momentum fade)
            if kama_dir == 1 or rsi_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_KAMA_RSI_Volume_Filter"
timeframe = "12h"
leverage = 1.0