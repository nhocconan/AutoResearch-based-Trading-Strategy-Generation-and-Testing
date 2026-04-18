#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and 1w RSI regime filter.
Go long when price breaks above Donchian upper band in low volatility (ATR ratio < 0.8) and bullish regime (weekly RSI > 50).
Go short when price breaks below Donchian lower band in low volatility and bearish regime (weekly RSI < 50).
Uses volatility filter to avoid whipsaws and regime filter to align with higher timeframe trend.
Designed for ~20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(len(high), np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.full(len(close), np.nan)
    rsi = np.full(len(close), np.nan)
    
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR(14)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for RSI(14)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) on 1d
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate RSI(14) on 1w
    rsi_14_1w = calculate_rsi(close_1w, 14)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_up = np.full(n, np.nan)
    donchian_down = np.full(n, np.nan)
    for i in range(20, n):
        donchian_up[i] = np.max(high[i-20:i])
        donchian_down[i] = np.min(low[i-20:i])
    
    # Calculate ATR ratio: current 14-period ATR / 50-period SMA of ATR
    atr_sma_50 = np.full(n, np.nan)
    for i in range(50, n):
        atr_sma_50[i] = np.mean(atr_14_1d[i-50:i])
    atr_ratio = np.full(n, np.nan)
    for i in range(50, n):
        if atr_sma_50[i] != 0:
            atr_ratio[i] = atr_14_1d[i] / atr_sma_50[i]
    
    # Align to 4h timeframe
    atr_14_1d_4h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    rsi_14_1w_4h = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    atr_ratio_4h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need ATR ratio calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_up[i]) or np.isnan(donchian_down[i]) or 
            np.isnan(atr_14_1d_4h[i]) or np.isnan(rsi_14_1w_4h[i]) or 
            np.isnan(atr_ratio_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: low volatility (ATR ratio < 0.8)
        vol_filter = atr_ratio_4h[i] < 0.8
        
        if position == 0:
            # Long: price breaks above Donchian upper, bullish regime, low volatility
            if close[i] > donchian_up[i] and rsi_14_1w_4h[i] > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, bearish regime, low volatility
            elif close[i] < donchian_down[i] and rsi_14_1w_4h[i] < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or volatility increases
            if close[i] < donchian_down[i] or atr_ratio_4h[i] >= 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or volatility increases
            if close[i] > donchian_up[i] or atr_ratio_4h[i] >= 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRFilter_1wRSI"
timeframe = "4h"
leverage = 1.0