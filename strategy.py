#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d RSI filter and volume confirmation. Long when KAMA rising AND 1d RSI > 50 AND volume > 1.5x 20-period average. Short when KAMA falling AND 1d RSI < 50 AND volume > 1.5x 20-period average. Exit when KAMA direction reverses.
# Uses adaptive trend (KAMA) to catch trends while avoiding whipsaws, with RSI filter to avoid counter-trend trades in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "12h_KAMA_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h KAMA (adaptive moving average)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    volatility = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_diff = np.diff(kama)
    kama_dir = np.where(kama_diff > 0, 1, np.where(kama_diff < 0, -1, 0))
    kama_dir = np.concatenate([[0], kama_dir])  # align with close
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 1d data
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.concatenate([np.full(14, np.nan), [np.mean(gain[:14])]])
    avg_loss = np.concatenate([np.full(14, np.nan), [np.mean(loss[:14])]])
    
    # Wilder smoothing
    for i in range(15, len(close_1d)):
        avg_gain = np.append(avg_gain, (avg_gain[-1] * 13 + gain[i-1]) / 14)
        avg_loss = np.append(avg_loss, (avg_loss[-1] * 13 + loss[i-1]) / 14)
    
    # Trim to match close_1d length
    avg_gain = avg_gain[14:]
    avg_loss = avg_loss[14:]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_dir[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA rising, RSI > 50, volume spike
            long_cond = (kama_dir[i] == 1) and (rsi_aligned[i] > 50) and volume_filter[i]
            # Short conditions: KAMA falling, RSI < 50, volume spike
            short_cond = (kama_dir[i] == -1) and (rsi_aligned[i] < 50) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns falling
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns rising
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals