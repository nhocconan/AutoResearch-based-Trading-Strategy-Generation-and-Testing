#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Filter_Volume_v2
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in choppy markets.
# Long when price > KAMA and RSI > 50, short when price < KAMA and RSI < 50.
# Filtered by 1w trend (close > 1w EMA20) and volume > 1.5x average volume.
# Designed for low frequency (10-25 trades/year) to avoid fee drag.
# Works in both bull and bear markets via adaptive trend filter and volume confirmation.

name = "1d_KAMA_Trend_RSI_Filter_Volume_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    if n < er_length:
        return kama
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros(n)
    for i in range(er_length, n):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Initialize KAMA
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    rsi = calculate_rsi(close, period=14)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2/21) + (ema_20_1w[i-1] * 19/21)
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma[i] = (volume[i] + vol_ma[i-1] * 19) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below weekly EMA20
        price_above_ema = close[i] > ema_20_1w_aligned[i]
        price_below_ema = close[i] < ema_20_1w_aligned[i]
        
        # KAMA and RSI conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        
        if position == 0:
            # LONG: price > KAMA, RSI > 50, price > weekly EMA20, volume filter
            if price_above_kama and rsi_above_50 and price_above_ema and volume_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA, RSI < 50, price < weekly EMA20, volume filter
            elif price_below_kama and rsi_below_50 and price_below_ema and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price < KAMA OR RSI < 50
            if price_below_kama or rsi_below_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > KAMA OR RSI > 50
            if price_above_kama or rsi_above_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals