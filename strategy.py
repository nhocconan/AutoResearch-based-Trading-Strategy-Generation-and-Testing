#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA with 1w EMA50 trend filter and volume confirmation
# KAMA adapts to market noise, reducing whipsaws in sideways markets
# 1w EMA50 provides strong trend bias to avoid counter-trend trades
# Volume confirmation filters weak breakouts and confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
name = "12h_KAMA_1wEMA50_Volume"
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
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA calculation
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    abs_change = np.abs(np.diff(close, n=1))
    er = np.zeros(n)
    for i in range(er_length, n):
        if np.sum(abs_change[i-er_length+1:i+1]) > 0:
            er[i] = change[i] / np.sum(abs_change[i-er_length+1:i+1])
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_length, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA + above 1w EMA50 + volume confirmation
            if (close[i] > kama[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + below 1w EMA50 + volume confirmation
            elif (close[i] < kama[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or below 1w EMA50
            if (close[i] < kama[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or above 1w EMA50
            if (close[i] > kama[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals