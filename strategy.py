#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d KAMA direction filter and RSI mean-reversion.
# Long: KAMA rising (bullish trend) + RSI < 30 (oversold) + volume > 1.5x avg volume.
# Short: KAMA falling (bearish trend) + RSI > 70 (overbought) + volume > 1.5x avg volume.
# Uses 1d KAMA for trend direction, 12h for RSI and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(change)
    if volatility > 0:
        er = np.abs(close_1d[-1] - close_1d[0]) / volatility
    else:
        er = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.full(len(close_1d), np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        change = np.abs(close_1d[i] - close_1d[i-1])
        volatility = np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1])) if i >= 1 else np.abs(close_1d[i] - close_1d[i-1]))
        if volatility > 0:
            er = np.abs(close_1d[i] - close_1d[0]) / volatility if i > 0 else 0
        else:
            er = 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close_1d[i] - kama[i-1])
    
    # KAMA direction: 1 = rising, -1 = falling
    kama_dir = np.zeros(len(close_1d))
    for i in range(1, len(kama)):
        if kama[i] > kama[i-1]:
            kama_dir[i] = 1
        elif kama[i] < kama[i-1]:
            kama_dir[i] = -1
        else:
            kama_dir[i] = kama_dir[i-1]
    
    # RSI (14-period) on 12h
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rsi = np.full(n, np.nan)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d KAMA direction to 12h
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        kama_dir_val = kama_dir_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: KAMA rising + RSI < 30 (oversold) + volume confirmation
            if (kama_dir_val == 1 and rsi_val < 30 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: KAMA falling + RSI > 70 (overbought) + volume confirmation
            elif (kama_dir_val == -1 and rsi_val > 70 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 70 (overbought) or KAMA turns bearish
            if rsi_val > 70 or kama_dir_val == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or KAMA turns bullish
            if rsi_val < 30 or kama_dir_val == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_Direction_RSI_MeanReversion"
timeframe = "12h"
leverage = 1.0