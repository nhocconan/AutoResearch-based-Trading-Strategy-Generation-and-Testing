#!/usr/bin/env python3
"""
Hypothesis: 12h 1-day KAMA trend with 1d RSI filter and 1d volume confirmation.
Uses KAMA (Kaufman Adaptive Moving Average) for trend direction on 12h timeframe,
RSI(14) > 50 for long and < 50 for short on daily timeframe to filter momentum,
and daily volume > 1.5x 20-period average for confirmation. Designed to work in
both bull and bear markets by using adaptive trend filter and avoiding overextended entries.
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
    
    # Get 1d data for RSI and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    
    # Get 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate KAMA(10) - Kaufman Adaptive Moving Average
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, 10))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # using fast=2, slow=30
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align 1d indicators to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(kama_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: KAMA trend + RSI filter + volume confirmation
        kama_long = close[i] > kama_aligned[i]
        kama_short = close[i] < kama_aligned[i]
        rsi_long = rsi_1d_aligned[i] > 50
        rsi_short = rsi_1d_aligned[i] < 50
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        long_entry = kama_long and rsi_long and vol_confirm
        short_entry = kama_short and rsi_short and vol_confirm
        
        # Exit when trend changes (KAMA cross)
        exit_long = position == 1 and close[i] < kama_aligned[i]
        exit_short = position == -1 and close[i] > kama_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_kama_rsi_volume"
timeframe = "12h"
leverage = 1.0