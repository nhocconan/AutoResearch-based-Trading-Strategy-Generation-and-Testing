#!/usr/bin/env python3
"""
4h KAMA Direction with RSI Momentum and Chop Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. 
RSI confirms momentum strength, while Choppiness Index filters ranging markets.
Works in bull markets via momentum continuation and in bear markets via mean reversion during low volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    if len(close) < er_length:
        return np.full_like(close, np.nan)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_length))
    change[0:er_length] = change[er_length] if er_length < len(close) else 0
    
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    volatility_arr = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i < er_length:
            volatility_arr[i] = np.sum(np.abs(np.diff(close[0:i+1]))) if i > 0 else 0
        else:
            volatility_arr[i] = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
    
    er = np.where(volatility_arr != 0, change / volatility_arr, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = np.zeros_like(high)
    for i in range(len(high)):
        if i < period:
            atr_sum[i] = np.sum(tr[0:i+1])
        else:
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    for i in range(len(high)):
        if i < period:
            highest_high[i] = np.max(high[0:i+1])
            lowest_low[i] = np.min(low[0:i+1])
        else:
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
    
    # Chop calculation
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 100 * np.log10(atr_sum / range_hl) / np.log10(period), 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily timeframe
    close_1d = df_1d['close'].values
    kama_1d = calculate_kama(close_1d, er_length=10, fast=2, slow=30)
    kama_1d_slope = np.diff(kama_1d, prepend=kama_1d[0])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    
    # Calculate RSI on daily timeframe
    rsi_1d = calculate_rsi(close_1d, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index on 4h timeframe
    chop = calculate_chop(high, low, close, period=14)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i > 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # KAMA slope positive = uptrend, negative = downtrend
        kama_up = kama_1d_aligned[i] > 0
        kama_down = kama_1d_aligned[i] < 0
        
        # RSI thresholds: avoid extremes, look for momentum
        rsi_momentum_up = rsi_1d_aligned[i] > 50
        rsi_momentum_down = rsi_1d_aligned[i] < 50
        
        # Chop filter: avoid strong trends (chop < 38.8) and extreme chop (chop > 61.8)
        chop_not_extreme = (chop[i] >= 38.8) and (chop[i] <= 61.8)
        
        if position == 0:
            # Long: KAMA up, RSI > 50, volume spike, not extreme chop
            if kama_up and rsi_momentum_up and vol_spike[i] and chop_not_extreme:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, volume spike, not extreme chop
            elif kama_down and rsi_momentum_down and vol_spike[i] and chop_not_extreme:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down or volatility dies
            if kama_down or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up or volatility dies
            if kama_up or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Chop_Volume"
timeframe = "4h"
leverage = 1.0