#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_R1S1_Breakout_Volume
Hypothesis: Use KAMA direction as primary trend filter on 1d to reduce whipsaws, combined with 1w KAMA trend confirmation and volume > 1.5x 20-day average. KAMA adapts to market noise, making it effective in both trending and ranging markets. Targets 15-25 trades/year by requiring alignment of 1d KAMA trend, price breakout beyond weekly KAMA, and volume confirmation. Works in bull markets by following uptrend breaks above weekly KAMA, and in bear markets by taking short breaks below weekly KAMA only when 1d KAMA confirms downtrend. Uses daily timeframe for higher win rate and lower trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) over 10 periods for 1d KAMA
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # sum of abs changes over 10 periods
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.665 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    
    # Align 1d KAMA to 1d timeframe (no alignment needed as we're on 1d)
    kama_1d_aligned = kama_1d  # already on 1d timeframe
    
    # Get 1w data for KAMA trend confirmation (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Efficiency Ratio (ER) over 10 periods for 1w KAMA
    change_1w = np.abs(np.diff(close_1w, n=10))
    volatility_1w = np.sum(np.abs(np.diff(close_1w, n=1)), axis=1)
    er_1w = np.zeros_like(close_1w)
    er_1w[10:] = change_1w[10:] / np.where(volatility_1w[10:] == 0, 1, volatility_1w[10:])
    sc_1w = (er_1w * (0.665 - 0.0645) + 0.0645) ** 2
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[9] = close_1w[9]
    for i in range(10, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    # Align 1w KAMA to 1d timeframe (wait for weekly bar close)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
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
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above 1d KAMA (uptrend), 1w KAMA confirms uptrend, and volume confirmation
            if (close[i] > kama_1d_aligned[i] and 
                close[i] > kama_1w_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 1d KAMA (downtrend), 1w KAMA confirms downtrend, and volume confirmation
            elif (close[i] < kama_1d_aligned[i] and 
                  close[i] < kama_1w_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below 1d KAMA (trend change) or 1w KAMA turns down
            if (close[i] < kama_1d_aligned[i] or 
                close[i] < kama_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above 1d KAMA (trend change) or 1w KAMA turns up
            if (close[i] > kama_1d_aligned[i] or 
                close[i] > kama_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_Trend_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0