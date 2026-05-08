#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Choppiness regime filter
# Long when KAMA direction is bullish (current > prior), RSI < 40 (oversold), and Chop < 38.2 (trending)
# Short when KAMA direction is bearish (current < prior), RSI > 60 (overbought), and Chop < 38.2 (trending)
# Exit when KAMA direction reverses
# Uses KAMA for adaptive trend, RSI for mean reversion entry, Chop to avoid ranging markets
# Target: 30-100 total trades over 4 years (7-25/year) for low fee drag

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w data for Chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_vals = np.full_like(price, np.nan, dtype=float)
        kama_vals[period] = price[period]
        for i in range(period+1, len(price)):
            if not np.isnan(kama_vals[i-1]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (price[i] - kama_vals[i-1])
            else:
                kama_vals[i] = price[i]
        return kama_vals
    
    # Calculate RSI
    def rsi(price, period=14):
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
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Calculate Choppiness Index
    def choppiness(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        # Chop calculation
        chop = np.full_like(close, np.nan, dtype=float)
        for i in range(period, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    # Calculate indicators
    kama_vals = kama(close, period=10, fast=2, slow=30)
    rsi_vals = rsi(close, period=14)
    chop_vals = choppiness(high, low, close, period=14)
    
    # Align Chop from weekly to daily
    chop_1d = align_htf_to_ltf(prices, df_1w, chop_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA bullish, RSI oversold, trending market
            long_cond = (kama_vals[i] > kama_vals[i-1]) and (rsi_vals[i] < 40) and (chop_1d[i] < 38.2)
            # Short conditions: KAMA bearish, RSI overbought, trending market
            short_cond = (kama_vals[i] < kama_vals[i-1]) and (rsi_vals[i] > 60) and (chop_1d[i] < 38.2)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns bearish
            if kama_vals[i] < kama_vals[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns bullish
            if kama_vals[i] > kama_vals[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals