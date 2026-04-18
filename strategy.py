#!/usr/bin/env python3
"""
4h KAMA + RSI + Chop Filter (Low-Frequency Trend Follower)
Hypothesis: KAMA adapts to market noise, reducing false signals in chop. 
RSI confirms momentum strength. Chop filter ensures trending conditions.
Designed for low trade frequency (<30/year) to avoid fee drag. Works in bull 
by catching trends and in bear by avoiding whipsaws via chop filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if n >= er_period else np.array([])
    if len(volatility) < n - er_period + 1:
        volatility = np.concatenate([np.full(er_period-1, np.nan), 
                                   [np.sum(np.abs(np.diff(close[i-er_period+1:i+1])) 
                                        for i in range(er_period-1, n))]])
    
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.full(n, np.nan)
    kama[er_period-1] = close[er_period-1]
    for i in range(er_period, n):
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

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate chop using daily ATR and range
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    true_range = np.maximum(high_1d - low_1d,
                           np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                     np.abs(low_1d - np.roll(close_1d, 1))))
    true_range[0] = high_1d[0] - low_1d[0]
    
    # Sum of true range over 14 periods
    sum_tr = np.zeros_like(atr_1d)
    for i in range(len(sum_tr)):
        if i < 14:
            sum_tr[i] = np.sum(true_range[max(0, i-13):i+1])
        else:
            sum_tr[i] = np.sum(true_range[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros_like(atr_1d)
    lowest_low = np.zeros_like(atr_1d)
    for i in range(len(highest_high)):
        if i < 14:
            highest_high[i] = np.max(high_1d[max(0, i-13):i+1])
            lowest_low[i] = np.min(low_1d[max(0, i-13):i+1])
        else:
            highest_high[i] = np.max(high_1d[i-13:i+1])
            lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    # Chop calculation
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = np.where(denominator != 0, 
                    100 * np.log10(sum_tr / denominator) / np.log10(14), 50)
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # KAMA on 4h close
    kama = calculate_kama(close, er_period=10, fast=2, slow=30)
    
    # RSI on 4h close
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
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is NOT choppy (< 61.8)
        trending = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, volume spike, trending market
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                vol_spike[i] and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, volume spike, trending market
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  vol_spike[i] and 
                  trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA or RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA or RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0