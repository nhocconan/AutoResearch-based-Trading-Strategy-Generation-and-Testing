#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day KAMA trend with RSI momentum filter and volume confirmation
# Long when 1-day KAMA is rising, RSI(14) > 50, and volume > 1.5x 20-period average
# Short when 1-day KAMA is falling, RSI(14) < 50, and volume > 1.5x 20-period average
# KAMA adapts to market noise, making it effective in both trending and ranging markets
# RSI filter ensures momentum alignment, volume confirmation reduces false signals
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "4h_1dKAMA_RSI_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day KAMA (adaptive moving average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 10:
            er[i] = np.nan
        else:
            direction = np.abs(close_1d[i] - close_1d[i-9])
            volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    # Smoothing constants
    sc = (er * 0.29 + 0.06) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # KAMA trend: slope over 3 periods
    kama_slope = np.diff(kama_aligned, n=1)
    kama_slope = np.insert(kama_slope, 0, np.nan)
    kama_rising = kama_slope > 0
    kama_falling = kama_slope < 0
    
    # RSI(14) on 4h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, volume confirmation
            if kama_rising[i] and rsi[i] > 50 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, volume confirmation
            elif kama_falling[i] and rsi[i] < 50 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling or RSI < 40
            if kama_falling[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising or RSI > 60
            if kama_rising[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals