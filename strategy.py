#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter_v3
Hypothesis: Use daily Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI for momentum filter, and Choppiness Index for regime filtering. Go long when KAMA is rising (bullish trend), RSI > 50 (bullish momentum), and market is trending (CHOP < 38.2). Short when KAMA is falling (bearish trend), RSI < 50 (bearish momentum), and market is trending. Weekly trend filter (price > weekly SMA50) ensures alignment with higher timeframe trend. Designed for 1d timeframe to limit trades and avoid fee drag, with weekly trend filter to work in both bull and bear markets.
"""

name = "1d_KAMA_Direction_RSI_Chop_Filter_v3"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate KAMA (trend direction)
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        if len(close_prices) < length:
            return np.full_like(close_prices, np.nan, dtype=float)
        change = np.abs(close_prices - np.roll(close_prices, length))
        change[:length] = 0  # First 'length' values have no change
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        # Handle volatility calculation properly
        volatility_series = pd.Series(close_prices).diff().abs()
        volatility = volatility_series.rolling(window=length, min_periods=1).sum().values
        volatility = np.where(volatility == 0, 1e-10, volatility)  # Avoid division by zero
        er = change / volatility
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close_prices, np.nan, dtype=float)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def calculate_rsi(close_prices, length=14):
        if len(close_prices) < length + 1:
            return np.full_like(close_prices, np.nan, dtype=float)
        delta = np.diff(close_prices)
        delta = np.concatenate([[0], delta])  # Prepend 0 for first element
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss == 0, 1e10, avg_gain / avg_loss)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index
    def calculate_choppiness(high_prices, low_prices, close_prices, length=14):
        if len(close_prices) < length:
            return np.full_like(close_prices, np.nan, dtype=float)
        atr = np.maximum(np.maximum(high_prices - low_prices, 
                                   np.abs(high_prices - np.roll(close_prices, 1))),
                        np.abs(low_prices - np.roll(close_prices, 1)))
        atr[0] = high_prices[0] - low_prices[0]  # First ATR value
        atr_sum = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
        highest_high = pd.Series(high_prices).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low_prices).rolling(window=length, min_periods=length).min().values
        range_max_min = highest_high - lowest_low
        range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)  # Avoid division by zero
        chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(length)
        return chop
    
    # Get daily data for indicator calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate indicators on daily data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    kama = calculate_kama(close_1d, length=10, fast=2, slow=30)
    rsi = calculate_rsi(close_1d, length=14)
    chop = calculate_choppiness(high_1d, low_1d, close_1d, length=14)
    
    # Align indicators to 1d timeframe (no extra delay needed as they're based on same timeframe)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get weekly SMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(sma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (bullish trend), RSI > 50 (bullish momentum), CHOP < 38.2 (trending market)
            if (kama_aligned[i] > kama_aligned[i-1] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 38.2 and 
                close[i] > sma_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (bearish trend), RSI < 50 (bearish momentum), CHOP < 38.2 (trending market)
            elif (kama_aligned[i] < kama_aligned[i-1] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 38.2 and 
                  close[i] < sma_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or RSI < 50 or CHOP >= 38.2 (ranging market) or price breaks below weekly SMA50
            if (kama_aligned[i] < kama_aligned[i-1] or 
                rsi_aligned[i] < 50 or 
                chop_aligned[i] >= 38.2 or 
                close[i] < sma_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or RSI > 50 or CHOP >= 38.2 (ranging market) or price breaks above weekly SMA50
            if (kama_aligned[i] > kama_aligned[i-1] or 
                rsi_aligned[i] > 50 or 
                chop_aligned[i] >= 38.2 or 
                close[i] > sma_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals