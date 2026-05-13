#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_TrendFilter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) direction on daily timeframe to capture trend, combined with RSI(14) for momentum confirmation and volume spike filter. Go long when KAMA is rising, RSI > 50, and volume > 1.5x 20-day average; short when KAMA is falling, RSI < 50, and volume spike. This filters false signals in choppy markets and works in both bull (catching trends) and bear (catching reversals) by aligning with adaptive trend and momentum.
"""

name = "1d_KAMA_Direction_RSI_TrendFilter"
timeframe = "1d"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER (Efficiency Ratio) = |change| / sum(|changes|) over period
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(change.reshape(-1, 1), axis=0)  # placeholder, will compute properly
    
    # Proper ER calculation
    diff = np.diff(close_1d)
    abs_diff = np.abs(diff)
    change_over_period = np.abs(np.diff(close_1d, 10))  # 10-period net change
    sum_abs_diff = np.convolve(abs_diff, np.ones(10), mode='same')
    sum_abs_diff[:9] = np.sum(abs_diff[:10])  # adjust for edges
    
    # Avoid division by zero
    er = np.where(sum_abs_diff > 0, change_over_period / sum_abs_diff, 0)
    # Smooth ER with smoothing constants
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # start at index 9
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no extra delay needed)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='same')
    # Handle first values
    avg_gain[:13] = np.nan
    avg_loss[:13] = np.nan
    # Wilder smoothing
    for i in range(13, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(rs == 0, 100, rsi)
    rsi = np.where(rs == np.inf, 0, rsi)
    
    # Align RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate volume average (20-day) for volume spike filter
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:19] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_aligned[i-1]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: KAMA rising (bullish trend), RSI > 50 (bullish momentum), volume spike
            if kama_aligned[i] > kama_aligned[i-1] and rsi_aligned[i] > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (bearish trend), RSI < 50 (bearish momentum), volume spike
            elif kama_aligned[i] < kama_aligned[i-1] and rsi_aligned[i] < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or RSI < 50
            if kama_aligned[i] < kama_aligned[i-1] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or RSI > 50
            if kama_aligned[i] > kama_aligned[i-1] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals