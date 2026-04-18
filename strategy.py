#!/usr/bin/env python3
"""
Hypothesis: 6h Hull Moving Average (HMA) crossover with 1-week RSI filter and volume confirmation.
- Long: HMA(20) crosses above HMA(50) AND weekly RSI(14) > 50 AND volume > 1.5x average
- Short: HMA(20) crosses below HMA(50) AND weekly RSI(14) < 50 AND volume > 1.5x average
- Exit: opposite HMA crossover
- Uses weekly RSI to filter for trend alignment on higher timeframe, reducing whipsaws in both bull and bear markets.
Designed for 12-37 trades/year (50-150 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.full(len(close), np.nan)
    for i in range(half_period - 1, len(close)):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.dot(close[i - half_period + 1:i + 1], weights) / weights.sum()
    
    # WMA of full period
    wma_full = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.dot(close[i - period + 1:i + 1], weights) / weights.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    hma_raw = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    hma = np.full(len(close), np.nan)
    for i in range(sqrt_period - 1, len(hma_raw)):
        if not np.isnan(hma_raw[i - sqrt_period + 1:i + 1]).any():
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.dot(hma_raw[i - sqrt_period + 1:i + 1], weights) / weights.sum()
    
    return hma

def calculate_rsi(close, period):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    # Initial average
    if len(gain) >= period:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
    
    # Wilder's smoothing
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if rs[i] is not np.nan:
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    rsi_14_1w = calculate_rsi(close_1w, 14)
    
    # Align weekly RSI to 6h timeframe
    rsi_14_1w_6h = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate HMA(20) and HMA(50) on 6h
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need HMA(50) and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(hma_20[i]) or np.isnan(hma_50[i]) or 
            np.isnan(rsi_14_1w_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # HMA crossover signals
        hma_cross_up = hma_20[i] > hma_50[i] and hma_20[i-1] <= hma_50[i-1]
        hma_cross_down = hma_20[i] < hma_50[i] and hma_20[i-1] >= hma_50[i-1]
        
        if position == 0:
            # Long: HMA bullish crossover, weekly RSI > 50, volume confirmation
            if hma_cross_up and rsi_14_1w_6h[i] > 50 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: HMA bearish crossover, weekly RSI < 50, volume confirmation
            elif hma_cross_down and rsi_14_1w_6h[i] < 50 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: HMA bearish crossover
            if hma_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: HMA bullish crossover
            if hma_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HMA20_50_RSI14_1w_Volume"
timeframe = "6h"
leverage = 1.0