#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_20_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a dynamic trend filter.
In trending markets, KAMA follows price closely; in ranging markets, it smooths out noise.
Combined with RSI(20) to avoid overextended entries and volume confirmation for breakout strength.
Designed for low trade frequency (target: 20-50/year) with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA with ER=10, FAST=2, SLOW=30
    kama_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 30:
        kama_1d[29] = close_1d[29]  # Initialize with price
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        for i in range(30, len(close_1d)):
            # Efficiency Ratio
            change = abs(close_1d[i] - close_1d[i-10])
            volatility = 0
            for j in range(1, 11):
                if i - j >= 0:
                    volatility += abs(close_1d[i-j] - close_1d[i-j-1])
            if volatility > 0:
                er = change / volatility
            else:
                er = 0
            # Smoothing Constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            # KAMA
            kama_1d[i] = kama_1d[i-1] + sc * (close_1d[i] - kama_1d[i-1])
    
    # Align daily KAMA to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(20) on 4h
    rsi = np.full(n, np.nan)
    if n >= 20:
        # Calculate price changes
        delta = np.diff(close)
        # Seed the first average
        gain = np.zeros(n)
        loss = np.zeros(n)
        gain[0] = max(delta[0], 0)
        loss[0] = max(-delta[0], 0)
        for i in range(1, n):
            gain[i] = max(delta[i], 0)
            loss[i] = max(-delta[i], 0)
        # Wilder's smoothing
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[19] = np.mean(gain[0:20])
        avg_loss[19] = np.mean(loss[0:20])
        for i in range(20, n):
            avg_gain[i] = (avg_gain[i-1] * 19 + gain[i]) / 20
            avg_loss[i] = (avg_loss[i-1] * 19 + loss[i]) / 20
        # Calculate RSI
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with RSI < 70 and volume spike
            if (close[i] > kama_1d_aligned[i] and rsi[i] < 70 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with RSI > 30 and volume spike
            elif (close[i] < kama_1d_aligned[i] and rsi[i] > 30 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or RSI > 70 (overbought)
            if (close[i] < kama_1d_aligned[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or RSI < 30 (oversold)
            if (close[i] > kama_1d_aligned[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_RSI_20_Filter"
timeframe = "4h"
leverage = 1.0