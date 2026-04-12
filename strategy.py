#!/usr/bin/env python3
# 12h_1d_kama_rsi_chop_v1
# Hypothesis: 12-hour strategy using KAMA trend direction, RSI overbought/oversold levels, and Choppiness Index regime filter.
# KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI provides mean-reversion signals.
# Choppiness Index filters trades: only take mean-reversion signals in high-chop (ranging) markets.
# Works in bull/bear by adapting to market conditions via KAMA and avoiding trend-following in chop.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "12h_1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on 12h close
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Calculate efficiency ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama = np.full_like(close, np.nan, dtype=float)
        kama[length] = close[length]
        for i in range(length + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # RSI on 1d close
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close_1d, length=14)
    
    # Choppiness Index on 1d data
    def calculate_choppiness(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(
                high[i] - low[i],
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            )
        # Sum of true range over period
        sum_atr = np.zeros_like(close)
        for i in range(length, len(close)):
            sum_atr[i] = np.sum(atr[i-length+1:i+1])
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        # Choppiness formula
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(length)
            else:
                chop[i] = 50  # Avoid division by zero
        return chop
    
    chop = calculate_choppiness(high_1d, low_1d, close_1d, length=14)
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # Using 1d data for alignment base
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in high-chop (ranging) markets: Choppiness > 61.8
        if chop_aligned[i] > 61.8:
            # Long signal: RSI oversold (< 30) and price above KAMA (bullish bias)
            if rsi_aligned[i] < 30 and close[i] > kama_aligned[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short signal: RSI overbought (> 70) and price below KAMA (bearish bias)
            elif rsi_aligned[i] > 70 and close[i] < kama_aligned[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit: RSI returns to neutral zone (40-60)
            elif position == 1 and rsi_aligned[i] > 40:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_aligned[i] < 60:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # In trending markets (low chop), stay flat to avoid whipsaws
            position = 0
            signals[i] = 0.0
    
    return signals