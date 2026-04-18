#!/usr/bin/env python3
"""
1h_4h_1d_KAMA_Trend_R1S1_Breakout_Volume
Hypothesis: Use 1h for entry timing, with 4h KAMA trend filter and 1d Camarilla R1/S1 breakout signals. 
This multi-timeframe approach filters noise: only take longs when 4h KAMA confirms uptrend and price breaks above 1d R1 with volume,
and shorts when 4h KAMA confirms downtrend and price breaks below 1d S1 with volume. 
Target: 15-35 trades/year by requiring alignment of higher timeframe trend, daily breakout, and volume confirmation.
Works in bull markets by following uptrend breaks above R1, and in bear markets by taking short breaks below S1 only when 4h KAMA confirms downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla R1 and S1
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Align levels to 1h timeframe (wait for bar close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 4h KAMA for adaptive trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_4h, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_4h, n=1)), axis=1)  # sum of abs changes over 10 periods
    # Avoid division by zero
    er = np.zeros_like(close_4h)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.665 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close_4h, np.nan)
    kama[9] = close_4h[9]  # seed
    for i in range(10, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Align KAMA to 1h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA and KAMA seeded
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(kama_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d R1, with volume, and 4h KAMA uptrend (price > KAMA)
            if (close[i] > r1_1d_aligned[i] and vol_confirm[i] and 
                close[i] > kama_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 1d S1, with volume, and 4h KAMA downtrend (price < KAMA)
            elif (close[i] < s1_1d_aligned[i] and vol_confirm[i] and 
                  close[i] < kama_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below KAMA (trend change) or breaks below S1 (failed breakout)
            if (close[i] < kama_aligned[i] or 
                (not np.isnan(s1_1d_aligned[i]) and close[i] < s1_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns above KAMA (trend change) or breaks above R1 (failed breakout)
            if (close[i] > kama_aligned[i] or 
                (not np.isnan(r1_1d_aligned[i]) and close[i] > r1_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_KAMA_Trend_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0