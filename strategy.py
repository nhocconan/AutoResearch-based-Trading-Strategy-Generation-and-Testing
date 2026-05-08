#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + RSI(14) pullback + volume confirmation on 12h
# Long when KAMA rising, RSI < 40 (pullback), and volume > 1.5x 20-period avg
# Short when KAMA falling, RSI > 60 (pullback), and volume > 1.5x 20-period avg
# Exit when RSI crosses back to neutral zone (40-60)
# KAMA adapts to market noise, reducing whipsaw in sideways markets
# RSI pullback enters during temporary counter-trend moves within trend
# Volume confirms institutional participation
# Target: 60-100 total trades over 4 years (15-25/year)

name = "12h_KAMA_RSI_Pullback_Volume"
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
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # KAMA calculation (using close prices)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs correction
    # Recalculate volatility properly
    volatility = np.zeros_like(change)
    for i in range(len(change)):
        if i >= 10:
            volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.zeros_like(close, dtype=np.float64)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA rising, RSI < 40 (pullback), volume filter
            kama_rising = kama[i] > kama[i-1]
            rsi_pullback = rsi[i] < 40
            
            # Short conditions: KAMA falling, RSI > 60 (pullback), volume filter
            kama_falling = kama[i] < kama[i-1]
            rsi_pullback_short = rsi[i] > 60
            
            if kama_rising and rsi_pullback and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif kama_falling and rsi_pullback_short and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 40 or KAMA turns down
            if rsi[i] >= 40 or kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses back below 60 or KAMA turns up
            if rsi[i] <= 60 or kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals