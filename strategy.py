#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Go long when price breaks above daily Camarilla R1 level with volume > 1.5x average and daily close > daily EMA34.
Go short when price breaks below daily Camarilla S1 level with volume > 1.5x average and daily close < daily EMA34.
Exit when price re-enters the daily Camarilla H-L range (S1 to R1).
Uses daily trend filter to avoid counter-trend trades. Designed for 4h timeframe to target 20-50 trades/year.
Daily Camarilla levels provide daily support/resistance; volume confirms breakout strength.
Works in bull/bear markets by following the higher timeframe trend.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate daily EMA34 for trend filter (using HTF data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA with proper smoothing
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])  # Initialize with SMA
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily high, low, close
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate daily Camarilla levels for R1 and S1
        rng_1d = high_1d - low_1d
        r1_1d = close_1d + rng_1d * 1.1 / 12
        s1_1d = close_1d - rng_1d * 1.1 / 12
        
        # Align to 4h timeframe
        r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
        s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
        
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume confirmation and 1d uptrend
            if close[i] > r1_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume confirmation and 1d downtrend
            elif close[i] < s1_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price re-enters the daily H-L range (S1 to R1)
            if close[i] < r1_1d_aligned[i] and close[i] > s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price re-enters the daily H-L range (S1 to R1)
            if close[i] < r1_1d_aligned[i] and close[i] > s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals