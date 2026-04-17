#!/usr/bin/env python3
"""
12h_1d_KAMA_RSI_TrendFilter_V1
Strategy: 12h KAMA trend with 1d RSI filter and volume confirmation.
Long: KAMA rising (trend up) + RSI(14) > 50 + volume > 1.5x 20-period average
Short: KAMA falling (trend down) + RSI(14) < 50 + volume > 1.5x 20-period average
Exit: Opposite KAMA direction change
Position size: 0.25
Uses KAMA for adaptive trend, RSI for momentum filter, volume for confirmation.
Designed to work in both bull and bear markets by adapting to volatility.
Timeframe: 12h
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
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    rsi_period = 14
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    for i in range(rsi_period + 1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi_padded = np.full(len(df_1d), np.nan)
    rsi_padded[rsi_period:] = rsi[rsi_period:]
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_padded)
    
    # Calculate 12h KAMA
    kama_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    change = np.abs(np.diff(close, k=1))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 0 else np.array([])
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        if np.sum(np.abs(np.diff(close[i-kama_period:i+1]))) > 0:
            er[i] = np.abs(close[i] - close[i-kama_period]) / np.sum(np.abs(np.diff(close[i-kama_period:i+1])))
        else:
            er[i] = 0
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align 12h KAMA to 12h timeframe (trivial since same timeframe)
    kama_aligned = kama
    
    # Calculate 12h volume average (20-period)
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(volume_ma20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume aligned to 12h
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        volume_filter = vol_12h_current > (1.5 * volume_ma20_12h_aligned[i])
        
        # KAMA direction
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI filter
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + volume filter
            if kama_rising and rsi_above_50 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + volume filter
            elif kama_falling and rsi_below_50 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling (trend change)
            if kama_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising (trend change)
            if kama_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_KAMA_RSI_TrendFilter_V1"
timeframe = "12h"
leverage = 1.0