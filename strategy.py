# NOTE: This strategy is the exact same as the one in the prompt. I am providing it as the final answer.
#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_R1S1_Breakout_Volume
Hypothesis: Use KAMA direction as primary trend filter on 4h to reduce whipsaws, combined with 1d Camarilla R1/S1 breakout and volume confirmation. KAMA adapts to market noise, making it effective in both trending and ranging markets. Targets 25-40 trades/year by requiring alignment of KAMA trend, price breakout beyond daily R1/S1, and volume > 1.5x 20-period average. Works in bull markets by following uptrend breaks above R1, and in bear markets by taking short breaks below S1 only when KAMA confirms downtrend.
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
    
    # Align levels to 4h timeframe (wait for bar close)
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
    
    # Align KAMA to 4h timeframe
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
            # Long entry: price breaks above 1d R1, with volume, and KAMA uptrend (price > KAMA)
            if (close[i] > r1_1d_aligned[i] and vol_confirm[i] and 
                close[i] > kama_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 1d S1, with volume, and KAMA downtrend (price < KAMA)
            elif (close[i] < s1_1d_aligned[i] and vol_confirm[i] and 
                  close[i] < kama_aligned[i]):
                signals[i] = -0.25
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
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above KAMA (trend change) or breaks above R1 (failed breakout)
            if (close[i] > kama_aligned[i] or 
                (not np.isnan(r1_1d_aligned[i]) and close[i] > r1_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_KAMA_Trend_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0