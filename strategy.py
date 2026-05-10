#!/usr/bin/env python3
# 1H_4H_1D_HMA_Trend_Follow
# Hypothesis: Use 4h HMA(21) for trend direction, 1d EMA(50) for long-term filter, and 1h HMA(9) for entry timing.
# Long when 4h HMA>1d EMA50 and price crosses above 1h HMA(9); short when 4h HMA<1d EMA50 and price crosses below 1h HMA(9).
# Works in bull/bear by following 4h trend with 1d filter to avoid counter-trend trades. Target: 15-30 trades/year per symbol.

name = "1H_4H_1D_HMA_Trend_Follow"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    wma_half = np.array([wma(arr[i:i+half], half) if i+half <= len(arr) else np.nan for i in range(len(arr))])
    wma_full = np.array([wma(arr[i:i+period], period) if i+period <= len(arr) else np.nan for i in range(len(arr))])
    
    raw = 2 * wma_half - wma_full
    hma = np.array([wma(raw[i:i+sqrt], sqrt) if i+sqrt <= len(raw) else np.nan for i in range(len(raw))])
    
    # Pad to original length
    hma_padded = np.full_like(arr, np.nan)
    start_idx = period - 1
    if len(hma) > 0 and start_idx < len(hma_padded):
        end_idx = start_idx + len(hma)
        if end_idx <= len(hma_padded):
            hma_padded[start_idx:end_idx] = hma
    return hma_padded

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h and 1d data
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # 4h HMA(21) for trend
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # 1d EMA(50) for long-term filter
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h HMA(9) for entry timing
    hma_1h = calculate_hma(close, 9)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(hma_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend conditions
        bullish_4h = hma_4h_aligned[i] > ema_1d_aligned[i]
        bearish_4h = hma_4h_aligned[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Enter long: bullish 4h trend + price crosses above 1h HMA
            if bullish_4h and close[i] > hma_1h[i] and (i == 0 or close[i-1] <= hma_1h[i-1]):
                signals[i] = 0.20
                position = 1
            # Enter short: bearish 4h trend + price crosses below 1h HMA
            elif bearish_4h and close[i] < hma_1h[i] and (i == 0 or close[i-1] >= hma_1h[i-1]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: bearish 4h trend or price crosses below 1h HMA
            if bearish_4h or (close[i] < hma_1h[i] and close[i-1] >= hma_1h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: bullish 4h trend or price crosses above 1h HMA
            if bullish_4h or (close[i] > hma_1h[i] and close[i-1] <= hma_1h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals