#!/usr/bin/env python3
# 4h_1d_ThreeSignal_Confluence
# Hypothesis: Confluence of 1) 1d Trend (Hull MA), 2) 1d Momentum (RSI extremes), and 3) 4h Breakout (Donchian) filters noise and captures strong moves.
# Works in bull/bear: Hull MA adapts to trend, RSI avoids overextension, Donchian ensures breakout confirmation.
# Target: 20-40 trades/year (80-160 total) for low fee drag.

name = "4h_1d_ThreeSignal_Confluence"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hull_moving_average(arr, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / (window * (window + 1) / 2)
    
    wma_half = np.full(n, np.nan)
    wma_full = np.full(n, np.nan)
    
    for i in range(half - 1, n):
        wma_half[i] = wma(arr[i - half + 1:i + 1], half)
    for i in range(period - 1, n):
        wma_full[i] = wma(arr[i - period + 1:i + 1], period)
    
    raw = 2 * wma_half - wma_full
    hma = np.full(n, np.nan)
    
    for i in range(sqrt - 1, n):
        if not np.isnan(raw[i]):
            start_idx = i - sqrt + 1
            end_idx = i + 1
            hma[i] = wma(raw[start_idx:end_idx], sqrt)
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Hull MA (55-period) for trend
    close_1d = df_1d['close'].values
    hull_ma = hull_moving_average(close_1d, 55)
    hull_ma_aligned = align_htf_to_ltf(prices, df_1d, hull_ma)
    
    # 1d RSI (14-period) for momentum/overextension
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(len(gain)):
        if i == 0:
            avg_gain[i] = gain[i]
            avg_loss[i] = loss[i]
        elif i < 14:
            avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
            avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 4h Donchian channels (20-period) for breakout
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(low_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 19:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(55, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hull_ma_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Hull MA (uptrend) + RSI not overbought (<70) + breakout above Donchian high
            if (close[i] > hull_ma_aligned[i] and 
                rsi_aligned[i] < 70 and 
                close[i] > donchian_high_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below Hull MA (downtrend) + RSI not oversold (>30) + breakdown below Donchian low
            elif (close[i] < hull_ma_aligned[i] and 
                  rsi_aligned[i] > 30 and 
                  close[i] < donchian_low_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price below Hull MA (trend change) OR RSI overbought (>80) OR breakdown below Donchian low
            if (close[i] < hull_ma_aligned[i] or 
                rsi_aligned[i] > 80 or 
                close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price above Hull MA (trend change) OR RSI oversold (<20) OR breakout above Donchian high
            if (close[i] > hull_ma_aligned[i] or 
                rsi_aligned[i] < 20 or 
                close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals