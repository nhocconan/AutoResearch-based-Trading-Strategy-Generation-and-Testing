#/usr/bin/env python3
"""
12h_1w_1d_Camarilla_R1S1_Breakout_Volume
Hypothesis: Use weekly and daily Camarilla pivot levels to identify key support/resistance zones, combined with volume confirmation and a weekly KAMA trend filter to reduce whipsaws. Targets 15-30 trades/year by requiring alignment of KAMA trend, price breakout beyond weekly R1/S1 or daily R1/S1, and volume > 1.5x 30-period average. Works in bull markets by following uptrend breaks above weekly R1/daily R1, and in bear markets by taking short breaks below weekly S1/daily S1 only when KAMA confirms downtrend. Uses 12h primary timeframe to balance trade frequency and signal quality.
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
    
    # Get 1w data for weekly Camarilla levels and KAMA
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w Camarilla R1 and S1
    rng_1w = high_1w - low_1w
    r1_1w = close_1w + rng_1w * 1.1 / 12
    s1_1w = close_1w - rng_1w * 1.1 / 12
    
    # Calculate 1d Camarilla R1 and S1
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Align weekly levels to 12h timeframe (wait for bar close)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Align daily levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate weekly KAMA for trend filter
    change = np.abs(np.diff(close_1w, n=10))
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=1)
    er = np.zeros_like(close_1w)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    sc = (er * (0.665 - 0.0645) + 0.0645) ** 2
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[9] = close_1w[9]
    for i in range(10, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
    
    # Align weekly KAMA to 12h timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Volume confirmation: current volume > 1.5 x 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need volume MA and KAMA seeded
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(kama_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1 OR daily R1, with volume, and KAMA uptrend
            if ((close[i] > r1_1w_aligned[i] or close[i] > r1_1d_aligned[i]) and 
                vol_confirm[i] and 
                close[i] > kama_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1 OR daily S1, with volume, and KAMA downtrend
            elif ((close[i] < s1_1w_aligned[i] or close[i] < s1_1d_aligned[i]) and 
                  vol_confirm[i] and 
                  close[i] < kama_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below KAMA (trend change) or breaks below weekly S1 or daily S1
            if (close[i] < kama_1w_aligned[i] or 
                close[i] < s1_1w_aligned[i] or 
                close[i] < s1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above KAMA (trend change) or breaks above weekly R1 or daily R1
            if (close[i] > kama_1w_aligned[i] or 
                close[i] > r1_1w_aligned[i] or 
                close[i] > r1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0