#!/usr/bin/env python3
"""
#100800 - 4h_KAMA_Direction_RSI_Chop_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction, RSI for momentum strength, and Choppiness Index for regime filtering. In trending markets (Chop < 38.2), follow KAMA direction with RSI confirmation. In ranging markets (Chop > 61.8), fade extreme RSI moves. Designed to work in both bull and bear markets by adapting to market regime. Targets 25-40 trades/year to minimize fee drag.
"""

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
    
    # Get 1d data for Choppiness Index (higher timeframe for regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (adaptive moving average)
    def calculate_kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
        kama = np.full_like(price, np.nan, dtype=float)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Calculate RSI
    def calculate_rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Calculate Choppiness Index on daily timeframe
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(np.diff(high))
        tr2 = np.abs(np.diff(low))
        tr3 = np.abs(np.diff(close))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr[1:] = tr
        
        # True Range for each period
        tr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if tr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Align KAMA and RSI (already calculated on 4h)
    kama_aligned = kama  # already on 4h timeframe
    rsi_aligned = rsi    # already on 4h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop[i] < 38.2:  # Trending market
            # Follow KAMA direction with RSI confirmation
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        elif chop[i] > 61.8:  # Ranging market
            # Fade extreme RSI moves
            if rsi_aligned[i] > 70 and close[i] < kama_aligned[i]:
                signals[i] = -0.20  # Short on overbought
                position = -1
            elif rsi_aligned[i] < 30 and close[i] > kama_aligned[i]:
                signals[i] = 0.20   # Long on oversold
                position = 1
            else:
                # Return to neutral
                if position == 1 and rsi_aligned[i] >= 50:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and rsi_aligned[i] <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold position if still in extreme
                    if position == 1:
                        signals[i] = 0.20
                    elif position == -1:
                        signals[i] = -0.20
                    else:
                        signals[i] = 0.0
        else:  # Transition zone (38.2 <= Chop <= 61.8)
            # Reduce position size in transition
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50:
                signals[i] = 0.15
                position = 1
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50:
                signals[i] = -0.15
                position = -1
            else:
                # Hold or flatten
                if position == 1:
                    signals[i] = 0.15
                elif position == -1:
                    signals[i] = -0.15
                else:
                    signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0