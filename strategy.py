#!/usr/bin/env python3
"""
4h_KAMA_Adaptive_Trend_With_Volume_Confirm
Hypothesis: Uses Kaufman's Adaptive Moving Average (KAMA) to capture trend direction with adaptive smoothing, combined with volume confirmation and RSI filter to avoid false signals. Designed for low trade frequency (<30/year) to minimize fee drag while capturing sustained trends in both bull and bear markets.
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
    
    # Get 1d data for KAMA trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    kama = calculate_kama(close_1d, length=10, fast=2, slow=30)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI on 1d for overbought/oversold filter
    rsi_1d = calculate_rsi(close_1d, length=14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for KAMA, RSI, and volume MA
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_conf = vol_confirm[i]
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, volume confirmation
            if close[i] > kama_val and rsi_val < 70 and vol_conf:
                signals[i] = size
                position = 1
            # Short: price below KAMA, RSI not oversold, volume confirmation
            elif close[i] < kama_val and rsi_val > 30 and vol_conf:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if close[i] < kama_val or rsi_val > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if close[i] > kama_val or rsi_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

def calculate_rsi(prices, length=14):
    """Calculate Relative Strength Index"""
    if len(prices) < length + 1:
        return np.full_like(prices, np.nan)
    
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(prices, np.nan)
    avg_loss = np.full_like(prices, np.nan)
    
    avg_gain[length] = np.mean(gain[:length])
    avg_loss[length] = np.mean(loss[:length])
    
    for i in range(length + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain[i-1]) / length
        avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss[i-1]) / length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_kama(close, length=10, fast=2, slow=30):
    """Calculate Kaufman's Adaptive Moving Average"""
    if len(prices) < length:
        return np.full_like(close, np.nan)
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(close, n=length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    
    # Handle array shapes
    er = np.full_like(close, np.nan)
    for i in range(length, len(close)):
        if volatility[i-length] != 0:
            er[i] = change[i-length] / volatility[i-length]
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[length] = close[length]
    
    for i in range(length + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

name = "4h_KAMA_Adaptive_Trend_With_Volume_Confirm"
timeframe = "4h"
leverage = 1.0