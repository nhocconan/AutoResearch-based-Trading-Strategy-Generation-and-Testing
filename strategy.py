#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Volume_Spike_v1
KAMA (ER=10) direction + RSI(14) > 50 for long / < 50 for short + volume spike confirmation.
Trades only in direction of higher timeframe (1d) trend.
Target: 20-50 total trades over 4 years (5-12/year).
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
    
    # === KAMA(ER=10) ===
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility properly
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Spike (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1d KAMA trend filter ===
    df_1d = get_htf_data(prices, '1d')
    # Compute KAMA on 1d close
    change_1d = np.abs(df_1d['close'] - np.roll(df_1d['close'], 10))
    volatility_1d = np.zeros_like(df_1d['close'])
    for i in range(10, len(df_1d)):
        volatility_1d[i] = np.sum(np.abs(np.diff(df_1d['close'].values[i-10:i+1])))
    er_1d = np.where(volatility_1d != 0, change_1d / volatility_1d, 0)
    sc_1d = (er_1d * (0.6645 - 0.0645) + 0.0645) ** 2
    kama_1d = np.full_like(df_1d['close'], np.nan)
    kama_1d[9] = df_1d['close'].values[9]
    for i in range(10, len(df_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (df_1d['close'].values[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(kama_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA rising, RSI > 50, volume spike, price above 1d KAMA
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                volume_spike[i] and 
                close[i] > kama_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA falling, RSI < 50, volume spike, price below 1d KAMA
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  volume_spike[i] and 
                  close[i] < kama_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA falling OR RSI < 50
            if (kama[i] < kama[i-1] or 
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR RSI > 50
            if (kama[i] > kama[i-1] or 
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_RSI_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0