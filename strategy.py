#!/usr/bin/env python3
"""
12h KAMA Direction + RSI + Chop Regime Filter
KAMA adapts to market efficiency, reducing whipsaw in chop and trending in trends.
RSI(14) filters for momentum strength. Chop filter avoids false signals in low volatility.
Designed for 12h timeframe with 1d HTF for regime filtering. Target: 20-30 trades/year.
Works in bull markets (trend following) and bear markets (mean reversion in chop).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    if n < er_length:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    change = np.abs(close[er_length:] - close[:-er_length])
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Proper volatility calculation
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+er_length]))) 
                          for i in range(n - er_length + 1)])
    volatility = np.concatenate([np.full(er_length-1, np.nan), volatility])
    
    er = np.where(volatility != 0, change / volatility, 0)
    er = np.concatenate([np.full(er_length-1, np.nan), er])
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    sc = np.concatenate([np.full(er_length-1, np.nan), sc[er_length-1:]])
    
    # KAMA
    kama = np.full(n, np.nan)
    kama[er_length-1] = close[er_length-1]  # seed
    
    for i in range(er_length, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Initial average
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Wilder smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index."""
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Sum of True Range
    atr_sum = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(period-1, n):
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    
    # Chop calculation
    chop = np.full(n, np.nan)
    for i in range(period-1, n):
        if highest_high[i] != lowest_low[i] and not np.isnan(atr_sum[i]):
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate indicators
    kama = calculate_kama(close, 10, 2, 30)
    rsi = calculate_rsi(close, 14)
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need all indicators warmed up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1d_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Chop regime: Chop > 61.8 = ranging, Chop < 38.2 = trending
        ranging = chop_1d_12h[i] > 61.8
        trending = chop_1d_12h[i] < 38.2
        
        if position == 0:
            # Ranging market: mean reversion at extremes
            if ranging:
                # Long when price below KAMA and RSI oversold
                if close[i] < kama[i] and rsi[i] < 30 and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short when price above KAMA and RSI overbought
                elif close[i] > kama[i] and rsi[i] > 70 and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
            # Trending market: follow momentum
            else:
                # Long when price above KAMA and RSI rising
                if close[i] > kama[i] and rsi[i] > 50 and rsi[i] < 70 and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short when price below KAMA and RSI falling
                elif close[i] < kama[i] and rsi[i] < 50 and rsi[i] > 30 and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses above KAMA or RSI overbought
            if close[i] > kama[i] * 1.01 or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below KAMA or RSI oversold
            if close[i] < kama[i] * 0.99 or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0