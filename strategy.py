#!/usr/bin/env python3
"""
12h KAMA Direction + RSI + Chop Regime Filter
Hypothesis: KAMA adapts to volatility, providing reliable trend direction.
Combined with RSI for momentum and Chop regime filter to avoid whipsaws.
Works in both bull and bear markets by filtering trades to strong trends only.
Designed for low trade frequency (~12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for Chop regime filter (more stable than daily)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate Choppy Index (Ehlers): measures market choppiness
    def calculate_chop(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = high[0] - low[0]
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_max_min = max_high - min_low
        range_max_min = np.where(range_max_min == 0, 1e-10, range_max_min)
        
        chop = 100 * np.log10(atr / range_max_min) / np.log10(window)
        return chop
    
    chop_weekly = calculate_chop(high_weekly, low_weekly, close_weekly, 14)
    chop_weekly = np.where(np.isnan(chop_weekly), 50, chop_weekly)  # neutral if NaN
    
    # Chop regime: < 38.2 = trending, > 61.8 = ranging
    chop_trending = chop_weekly < 38.2
    
    # Load daily data for KAMA and RSI
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    
    # Kaufman Adaptive Moving Average (KAMA)
    def calculate_kama(close, window=10, fast=2, slow=30):
        change = np.abs(close - np.roll(close, window))
        change[0:window] = 0
        
        volatility = np.zeros_like(close)
        for i in range(1, len(close)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        volatility[0:window] = 0
        
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_daily = calculate_kama(close_daily, 10, 2, 30)
    
    # RSI
    def calculate_rsi(close, window=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean().values
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean().values
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_daily = calculate_rsi(close_daily, 14)
    
    # Align weekly and daily indicators to 12h timeframe
    chop_trending_aligned = align_htf_to_ltf(prices, df_weekly, chop_trending.astype(float))
    kama_daily_aligned = align_htf_to_ltf(prices, df_daily, kama_daily)
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(chop_trending_aligned[i]) or np.isnan(kama_daily_aligned[i]) or 
            np.isnan(rsi_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama = kama_daily_aligned[i]
        rsi = rsi_daily_aligned[i]
        chop_trend = chop_trending_aligned[i] > 0.5  # True if trending regime
        
        if position == 0:
            # Long entry: price > KAMA, RSI > 50, and trending regime
            if price > kama and rsi > 50 and chop_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA, RSI < 50, and trending regime
            elif price < kama and rsi < 50 and chop_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA or RSI < 40
            if price < kama or rsi < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA or RSI > 60
            if price > kama or rsi > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0