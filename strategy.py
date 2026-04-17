#!/usr/bin/env python3
"""
4h_KAMA_RSI_BBands_Volume_v2
KAMA direction + RSI + Bollinger Band squeeze + volume confirmation.
Trades only during expansion phases after low volatility (BB width < 20th percentile).
Uses 1d trend filter: price above/below 1d EMA100.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === KAMA(10, 2, 30) ===
    def kama(close, er_len=10, fast_sc=2, slow_sc=30):
        change = np.abs(close - np.roll(close, er_len))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close)
    kama_dir = np.where(kama_vals > np.roll(kama_vals, 1), 1, -1)
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Bollinger Bands Width(20, 2) ===
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_width = (bb_std * 4) / bb_mid  # (upper - lower) / mid
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=1).rank(pct=True).values
    
    # === Volume Ratio (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === 1d EMA100 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_dir[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(ema_100_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA up, RSI > 50, BB width expanding (above 20th percentile), volume > 1.5x, price above 1d EMA100
            if (kama_dir[i] == 1 and 
                rsi[i] > 50 and 
                bb_width_percentile[i] > 0.2 and 
                vol_ratio[i] > 1.5 and 
                close[i] > ema_100_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA down, RSI < 50, BB width expanding, volume > 1.5x, price below 1d EMA100
            elif (kama_dir[i] == -1 and 
                  rsi[i] < 50 and 
                  bb_width_percentile[i] > 0.2 and 
                  vol_ratio[i] > 1.5 and 
                  close[i] < ema_100_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA down OR RSI < 40
            if (kama_dir[i] == -1 or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA up OR RSI > 60
            if (kama_dir[i] == 1 or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_BBands_Volume_v2"
timeframe = "4h"
leverage = 1.0