#!/usr/bin/env python3
# 4h_1dKAMA_Trend_Filtered_By_RSI
# Uses 1-day KAMA for trend direction and 1-day RSI for overbought/oversold conditions
# with volume confirmation on 4h timeframe. KAMA adapts to market noise, reducing whipsaw
# in choppy markets while RSI prevents entries at extremes. Designed for low trade frequency
# (target 20-50/year) with 0.25 position sizing to work in both bull and bear markets.

name = "4h_1dKAMA_Trend_Filtered_By_RSI"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_ema=2, slow_ema=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), mode='valid') if len(close) > 1 else 0
    # Full volatility calculation using rolling sum
    volatility_full = np.zeros_like(close)
    for i in range(len(close)):
        if i == 0:
            volatility_full[i] = 0
        else:
            volatility_full[i] = np.sum(np.abs(np.diff(close[max(0, i-er_length+1):i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility_full[i] > 0:
            er[i] = change[i] / volatility_full[i]
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
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
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily KAMA for trend
    kama_1d = calculate_kama(close_1d, er_length=10, fast_ema=2, slow_ema=30)
    kama_4h = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate daily RSI for overbought/oversold
    rsi_1d = calculate_rsi(close_1d, period=14)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily volume filter (20-period MA)
    vol_ma_20_1d = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if i < 20:
            vol_ma_20_1d[i] = np.nan
        else:
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > (2.0 * vol_ma_20_4h)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI not overbought (< 70), volume spike
            if close[i] > kama_4h[i] and rsi_4h[i] < 70 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price below KAMA (downtrend), RSI not oversold (> 30), volume spike
            elif close[i] < kama_4h[i] and rsi_4h[i] > 30 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price crosses below KAMA or RSI overbought (> 80) or 5 bars elapsed
            if bars_since_entry >= 5 or close[i] < kama_4h[i] or rsi_4h[i] > 80:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or RSI oversold (< 20) or 5 bars elapsed
            if bars_since_entry >= 5 or close[i] > kama_4h[i] or rsi_4h[i] < 20:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals