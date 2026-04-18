#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA trend with RSI confirmation and volume spike filter.
- Long: KAMA rising, RSI > 50 and < 70, volume > 2x 20-period average
- Short: KAMA falling, RSI < 50 and > 30, volume > 2x 20-period average
- Exit: opposite KAMA direction or RSI extremes (RSI > 70 for long, RSI < 30 for short)
- Uses adaptive trend (KAMA) to avoid whipsaws in ranging markets, RSI for momentum filter.
Designed for 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    if len(close) < er_period:
        return np.full(len(close), np.nan)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros(len(close))
    er[er_period:] = change[er_period-1:] / volatility[er_period-1:]
    
    # Calculate Smoothing Constant
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (10-period ER, 2/30 SC)
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    
    # First average
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Subsequent averages
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.zeros(len(close))
    rs[14:] = avg_gain[14:] / np.where(avg_loss[14:] == 0, 1e-10, avg_loss[14:])
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need KAMA, RSI, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = i > 0 and kama[i] > kama[i-1]
        kama_falling = i > 0 and kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA rising, RSI between 50-70, volume confirmation
            if kama_rising and 50 < rsi[i] < 70 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI between 30-50, volume confirmation
            elif kama_falling and 30 < rsi[i] < 50 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling or RSI > 70 (overbought)
            if kama_falling or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising or RSI < 30 (oversold)
            if kama_rising or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Volume"
timeframe = "4h"
leverage = 1.0