#!/usr/bin/env python3
"""
12h_KAMA_Trend_1dRSI_VolumeFilter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 12h defines trend, with daily RSI for momentum confirmation and volume filter to avoid false breaks. Works in both bull and bear by following adaptive trend. Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # first 14 gains
    avg_loss[13] = np.mean(loss[1:14])  # first 14 losses
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan  # not enough data
    
    # Calculate volume average (20-period) on 1d
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align 1d indicators to 12h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 12h data for KAMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA(10,2,30) on 12h close
    close_12h = df_12h['close'].values
    fast_sc = 2/(2+1)      # EMA(2) smoothing constant
    slow_sc = 2/(30+1)     # EMA(30) smoothing constant
    
    # Efficiency ratio
    change = np.abs(np.diff(close_12h, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder - will compute properly below
    
    # Proper volatility calculation (sum of absolute changes over 10 periods)
    volatility = np.zeros_like(close_12h)
    for i in range(10, len(close_12h)):
        volatility[i] = np.sum(np.abs(np.diff(close_12h[i-10:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close_12h)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constant
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # start with close
    
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (no additional delay needed for trend)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(30, 30)  # RSI and KAMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(kama_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price relative to KAMA
        above_kama = price > kama_aligned[i]
        below_kama = price < kama_aligned[i]
        
        # Momentum filter: RSI not extreme
        rsi_momentum = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Volume filter: current volume > 1.2x daily average
        volume_filter = volume[i] > (vol_ma_1d_aligned[i] * 1.2)
        
        if position == 0:
            # Long: price above KAMA with momentum and volume
            if above_kama and rsi_momentum and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with momentum and volume
            elif below_kama and rsi_momentum and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below KAMA
            if below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_1dRSI_VolumeFilter"
timeframe = "12h"
leverage = 1.0