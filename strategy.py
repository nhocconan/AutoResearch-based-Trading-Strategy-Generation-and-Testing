#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_R1S1_Breakout_Volume
Hypothesis: Use KAMA direction as primary trend filter on daily timeframe to reduce whipsaws, combined with weekly KAMA trend confirmation and daily price breakout beyond weekly R1/S1 levels. Weekly R1/S1 are calculated from prior week's range. Weekly timeframe provides stronger trend context, reducing false breakouts. Targets 15-25 trades/year by requiring alignment of daily KAMA uptrend/downtrend, price breakout beyond weekly R1/S1, and volume > 1.5x 20-day average. Works in bull markets by following uptrend breaks above weekly R1, and in bear markets by taking short breaks below weekly S1 only when daily KAMA confirms downtrend.
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
    
    # Get weekly data for KAMA trend and R1/S1 levels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly KAMA (10-period)
    change = np.abs(np.diff(close_1w, n=10))
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=1)
    er = np.zeros_like(close_1w)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    sc = (er * (0.665 - 0.0645) + 0.0645) ** 2
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[9] = close_1w[9]
    for i in range(10, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate weekly R1 and S1 from prior week's range
    rng_1w = high_1w - low_1w
    r1_1w = close_1w + rng_1w * 1.1 / 12
    s1_1w = close_1w - rng_1w * 1.1 / 12
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get daily KAMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    change_d = np.abs(np.diff(close_1d, n=10))
    volatility_d = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)
    er_d = np.zeros_like(close_1d)
    er_d[10:] = change_d[10:] / np.where(volatility_d[10:] == 0, 1, volatility_d[10:])
    sc_d = (er_d * (0.665 - 0.0645) + 0.0645) ** 2
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[9] = close_1d[9]
    for i in range(10, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_d[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1, with volume, and both weekly and daily KAMA uptrend
            if (close[i] > r1_1w_aligned[i] and vol_confirm[i] and 
                close[i] > kama_1w_aligned[i] and close[i] > kama_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1, with volume, and both weekly and daily KAMA downtrend
            elif (close[i] < s1_1w_aligned[i] and vol_confirm[i] and 
                  close[i] < kama_1w_aligned[i] and close[i] < kama_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below weekly KAMA or daily KAMA (trend change) or breaks below weekly S1
            if (close[i] < kama_1w_aligned[i] or close[i] < kama_1d_aligned[i] or 
                (not np.isnan(s1_1w_aligned[i]) and close[i] < s1_1w_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly KAMA or daily KAMA (trend change) or breaks above weekly R1
            if (close[i] > kama_1w_aligned[i] or close[i] > kama_1d_aligned[i] or 
                (not np.isnan(r1_1w_aligned[i]) and close[i] > r1_1w_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_Trend_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0