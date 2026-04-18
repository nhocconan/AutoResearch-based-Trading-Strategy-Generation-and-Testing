#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_and_ChopFilter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h timeframe to detect trend direction, confirmed by volume spike (>1.8x 20-period average) and choppy market filter (Choppiness Index < 61.8 for trending regime). Enter long when price crosses above KAMA with volume confirmation in trending market, short when price crosses below KAMA with volume confirmation. Exit when price crosses back across KAMA. Designed for low-frequency trading (12-37 trades/year) to minimize fee decay while capturing sustained trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) calculation
    def calculate_kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama = np.full_like(price, np.nan, dtype=float)
        kama[period] = price[period]  # Initialize
        
        for i in range(period+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        
        return kama
    
    # Choppiness Index
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Sum of True Range over period
        atr_sum = np.nansum(tr, axis=1)  # Will be replaced with proper rolling sum
        
        # Proper rolling sum for ATR
        tr_series = pd.Series(tr)
        atr_sum = tr_series.rolling(window=period, min_periods=period).sum().values
        
        # Highest high and lowest low over period
        max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        # Chop calculation
        range_max_min = max_high - min_low
        chop = np.where(range_max_min != 0, 
                        100 * np.log10(atr_sum / range_max_min) / np.log10(period), 
                        50)
        return chop
    
    # Calculate KAMA on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    kama_12h = calculate_kama(close_12h, period=10, fast=2, slow=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate Choppiness Index on 1d timeframe for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama = kama_12h_aligned[i]
        chop = chop_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Only trade in trending markets (Chop < 61.8)
        is_trending = chop < 61.8
        
        if position == 0:
            # Long: price crosses above KAMA with volume spike in trending market
            if price > kama and is_trending and vol_spike:
                # Need to confirm crossover - price was below KAMA previously
                if i > start_idx and close[i-1] <= kama_12h_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
            # Short: price crosses below KAMA with volume spike in trending market
            elif price < kama and is_trending and vol_spike:
                # Need to confirm crossover - price was above KAMA previously
                if i > start_idx and close[i-1] >= kama_12h_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses back below KAMA
            if price < kama and close[i-1] >= kama_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses back above KAMA
            if price > kama and close[i-1] <= kama_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_With_Volume_and_ChopFilter"
timeframe = "12h"
leverage = 1.0