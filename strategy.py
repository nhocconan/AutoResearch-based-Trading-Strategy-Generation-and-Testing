#!/usr/bin/env python3
"""
12h_1w_rsi_divergence_with_volume
Uses RSI divergence on weekly timeframe to detect potential reversals in 12h timeframe.
Looks for bearish divergence (price makes higher high, RSI makes lower high) for shorts
and bullish divergence (price makes lower low, RSI makes higher low) for longs.
Requires volume confirmation and uses RSI extremes for entry filtering.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in both trending and ranging markets by capturing exhaustion points.
"""

name = "12h_1w_rsi_divergence_with_volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper handling"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Set initial values to 50 (neutral)
    rsi[:period] = 50
    return rsi

def find_divergence(price, rsi, lookback=10):
    """Find bullish and bearish divergence"""
    n = len(price)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Look for local extrema in the lookback window
        window_start = max(0, i - lookback)
        window_end = i
        
        # Find price low and high in window
        price_low_idx = np.argmin(price[window_start:window_end]) + window_start
        price_high_idx = np.argmax(price[window_start:window_end]) + window_start
        
        # Find RSI low and high in window
        rsi_low_idx = np.argmin(rsi[window_start:window_end]) + window_start
        rsi_high_idx = np.argmax(rsi[window_start:window_end]) + window_start
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (price[price_low_idx] < price[window_start] and 
            rsi[rsi_low_idx] > rsi[window_start]):
            bullish_div[i] = True
            
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (price[price_high_idx] > price[window_start] and 
            rsi[rsi_high_idx] < rsi[window_start]):
            bearish_div[i] = True
    
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI and divergence
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate RSI on weekly close
    rsi_1w = calculate_rsi(close_1w, 14)
    
    # Find divergences
    bullish_div, bearish_div = find_divergence(close_1w, rsi_1w, 10)
    
    # Align divergence signals to 12h
    bullish_div_aligned = align_htf_to_ltf(prices, df_1w, bullish_div.astype(float))
    bearish_div_aligned = align_htf_to_ltf(prices, df_1w, bearish_div.astype(float))
    
    # Volume confirmation on 12h: volume > 1.3x 20-period average
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:19] = np.nan  # Mark insufficient data
    vol_confirm = volume > (vol_ma * 1.3)
    
    # RSI extremes filter: avoid overbought/oversold extremes for better entries
    rsi_overbought = 70
    rsi_oversold = 30
    rsi_not_extreme = (rsi_1w < rsi_overbought) & (rsi_1w > rsi_oversold)
    rsi_not_extreme_aligned = align_htf_to_ltf(prices, df_1w, rsi_not_extreme.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bullish_div_aligned[i]) or np.isnan(bearish_div_aligned[i]) or 
            np.isnan(vol_confirm[i]) or np.isnan(rsi_not_extreme_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: bullish divergence on weekly + volume + not extreme RSI
        if (bullish_div_aligned[i] > 0.5 and vol_confirm[i] and 
            rsi_not_extreme_aligned[i] > 0.5 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: bearish divergence on weekly + volume + not extreme RSI
        elif (bearish_div_aligned[i] > 0.5 and vol_confirm[i] and 
              rsi_not_extreme_aligned[i] > 0.5 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite divergence or RSI extreme
        elif position == 1 and (bearish_div_aligned[i] > 0.5 or rsi_not_extreme_aligned[i] < 0.5):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_div_aligned[i] > 0.5 or rsi_not_extreme_aligned[i] < 0.5):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals