#!/usr/bin/env python3
"""
12h KAMA Direction with RSI Filter and Volume Confirmation
Hypothesis: KAMA adapts to market noise, providing smooth trend direction. 
Combined with RSI extremes and volume confirmation, it captures strong moves 
while avoiding chop. Works in both bull and bear markets by following 
adaptive trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    if len(close) < er_period + 1:
        return np.full_like(close, np.nan)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) >= er_period else np.array([np.sum(np.abs(np.diff(close)))])
    # For rolling volatility calculation
    volatility_rolling = np.zeros_like(close)
    for i in range(len(close)):
        if i < er_period:
            volatility_rolling[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period+1):i+1]))) if i >= 1 else 0
        else:
            volatility_rolling[i] = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
    
    er = np.where(volatility_rolling != 0, change / volatility_rolling, 0)
    
    # Smoothing constants
    sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA on 1d timeframe
    kama_1d = calculate_kama(df_1d['close'].values)
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI on 12h price
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, with volume
            if (close[i] > kama_aligned[i] and 
                rsi[i] < 70 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI not oversold, with volume
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] > 30 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if close[i] < kama_aligned[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if close[i] > kama_aligned[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_VolumeFilter"
timeframe = "12h"
leverage = 1.0