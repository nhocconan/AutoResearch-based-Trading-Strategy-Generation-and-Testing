#!/usr/bin/env python3
"""
12h_KAMA_Camarilla_Breakout_v1
Hypothesis: On 12h timeframe, using 1d KAMA as a dynamic trend filter combined with 1d Camarilla R3/S3 breakouts and volume confirmation reduces whipsaws in ranging markets while capturing strong trends. The 12h timeframe naturally limits trade frequency to avoid fee drag (target: 50-150 trades over 4 years). KAMA adapts to market noise, making it effective in both bull and bear regimes by avoiding false breakouts during choppy periods. Volume confirmation ensures breakouts have conviction. Target: 75-150 total trades over 4 years (19-37/year) with improved Sharpe via better trend alignment.
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
    
    # Load 1d data ONCE before loop for HTF trend filter (KAMA) and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for trend filter (adaptive to market noise)
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        net_change = np.abs(close_1d[i] - close_1d[i-10])
        total_change = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
        er[i] = net_change / total_change if total_change != 0 else 0
    # Smoothing constants: fastest SC = 2/(2+1) = 0.666, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.666 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed value
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d Camarilla levels (R3, S3) using previous 1d's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])  # previous 1d close
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d_shifted + 1.1 * camarilla_range / 4  # R3 level
    s3 = close_1d_shifted - 1.1 * camarilla_range / 4  # S3 level
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h volume confirmation: volume > 2.0x 20-period average (stricter for lower frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for KAMA and volume MA)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (KAMA)
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Volume confirmation (stricter threshold for 12h)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions (R3/S3 for stronger breakouts)
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Long logic: breakout above R3 in uptrend with volume
        if uptrend and volume_spike and breakout_r3:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S3 in downtrend with volume
        elif downtrend and volume_spike and breakout_s3:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Camarilla_Breakout_v1"
timeframe = "12h"
leverage = 1.0