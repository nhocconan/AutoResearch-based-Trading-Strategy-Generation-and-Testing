#!/usr/bin/env python3
"""
4h_KAMA_1dRSI_Trend_Filter
Strategy: Use 4h KAMA to determine trend direction, 1d RSI for overbought/oversold filter.
- Long when KAMA turns up (bullish) AND 1d RSI < 50 (not overbought)
- Short when KAMA turns down (bearish) AND 1d RSI > 50 (not oversold)
- Exit when KAMA reverses direction
- Uses volume confirmation (volume > 1.5x 20-period average)
- Designed for 20-40 trades/year per symbol
Works in bull markets by catching trends, in bear markets by avoiding overextended moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    if len(close) < er_length:
        return np.full(len(close), np.nan)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros(len(close))
    er[er_length-1:] = change / np.where(volatility[er_length-1:] == 0, 1, volatility[er_length-1:])
    
    # Smoothing Constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    rsi_1d = calculate_rsi(close_1d, period=14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Align 1d RSI to 4h timeframe
    rsi_1d_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need sufficient data for KAMA, RSI, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        # KAMA direction: compare current vs previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA turning up AND RSI not overbought (< 50) AND volume
            if kama_up and rsi_1d_4h[i] < 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down AND RSI not oversold (> 50) AND volume
            elif kama_down and rsi_1d_4h[i] > 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down
            if kama_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up
            if kama_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_1dRSI_Trend_Filter"
timeframe = "4h"
leverage = 1.0