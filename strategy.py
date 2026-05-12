#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Filter_Volume
# Hypothesis: Use KAMA to determine trend direction on 4h, filtered by RSI and volume spike.
# Enter long when KAMA shows upward trend, RSI < 70 (not overbought), and volume > 1.5x average.
# Enter short when KAMA shows downward trend, RSI > 30 (not oversold), and volume > 1.5x average.
# Exit when trend reverses or volume drops. Designed for 15-30 trades/year to avoid fee drag.
# Works in bull markets (catching trends) and bear markets (catching corrections/trends).

name = "4h_KAMA_Trend_RSI_Filter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_length=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average.
    Returns KAMA array.
    """
    n = len(close)
    if n == 0:
        return np.array([])
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close))
    
    # Sum of absolute changes over er_length period
    er_numerator = np.zeros(n)
    er_denominator = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - er_length + 1)
        er_numerator[i] = np.sum(change[start_idx:i+1])
        er_denominator[i] = np.sum(abs_change[start_idx:i+1])
    
    er = np.where(er_denominator > 0, er_numerator / er_denominator, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama_vals = np.zeros(n)
    kama_vals[0] = close[0]
    
    for i in range(1, n):
        kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
    
    return kama_vals

def rsi(close, length=14):
    """
    Relative Strength Index.
    Returns RSI array.
    """
    n = len(close)
    if n < 2:
        return np.full(n, 50.0)
    
    # Price changes
    delta = np.diff(close, prepend=close[0])
    
    # Gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Smoothed averages
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # Initial average
    avg_gain[length-1] = np.mean(gains[1:length]) if length > 1 else 0
    avg_loss[length-1] = np.mean(losses[1:length]) if length > 1 else 0
    
    # Wilder smoothing
    for i in range(length, n):
        avg_gain[i] = (avg_gain[i-1] * (length-1) + gains[i]) / length
        avg_loss[i] = (avg_loss[i-1] * (length-1) + losses[i]) / length
    
    # RSI calculation
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi_vals = 100 - (100 / (1 + rs))
    
    return rsi_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA for trend (10,2,30)
    kama_vals = kama(close, er_length=10, fast=2, slow=30)
    
    # RSI for momentum filter
    rsi_vals = rsi(close, length=14)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price relative to KAMA
        trend_up = close[i] > kama_vals[i]
        trend_down = close[i] < kama_vals[i]
        
        # RSI filter: not extreme
        rsi_not_overbought = rsi_vals[i] < 70
        rsi_not_oversold = rsi_vals[i] > 30
        
        # Volume filter: spike above average
        vol_spike = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # LONG: Upward trend, not overbought, volume spike
            if trend_up and rsi_not_overbought and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Downward trend, not oversold, volume spike
            elif trend_down and rsi_not_oversold and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend reverses or volume drops
            if not trend_up or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reverses or volume drops
            if not trend_down or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals