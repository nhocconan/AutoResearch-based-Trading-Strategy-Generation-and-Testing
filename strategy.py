#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: On 1d timeframe, use KAMA to determine trend direction (adaptive moving average),
# RSI for overbought/oversold conditions, and Choppiness Index to filter ranging markets.
# Enter long when KAMA turns up, RSI < 50, and Chop > 61.8 (ranging market - mean reversion).
# Enter short when KAMA turns down, RSI > 50, and Chop > 61.8.
# Exit when KAMA reverses direction or Chop < 38.2 (trending market).
# Uses weekly trend filter: only trade in direction of weekly KAMA.
# Designed for low-frequency trading (1d) to minimize fee drag and work in both bull and bear markets.

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(price, period=10, fast=2, slow=30):
        # Efficiency ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_vals = np.full_like(price, np.nan)
        kama_vals[period] = price[period]
        for i in range(period+1, len(price)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (price[i] - kama_vals[i-1])
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
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        rsi_vals[:period] = np.nan
        return rsi_vals
    
    # Calculate Choppiness Index
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First period
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # Avoid division by zero
        chop[:period-1] = np.nan
        return chop
    
    # Calculate indicators
    kama_vals = kama(close, period=10, fast=2, slow=30)
    rsi_vals = rsi(close, period=14)
    chop_vals = choppiness_index(high, low, close, period=14)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_kama = kama(weekly_close, period=10, fast=2, slow=30)
    weekly_kama_aligned = align_htf_to_ltf(prices, df_1w, weekly_kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_vals[i]) or np.isnan(weekly_kama_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama_vals[i]
        kama_prev = kama_vals[i-1]
        rsi_val = rsi_vals[i]
        chop_val = chop_vals[i]
        weekly_kama_val = weekly_kama_aligned[i]
        weekly_kama_prev = weekly_kama_aligned[i-1] if i > 0 else weekly_kama_val
        
        # Determine KAMA direction
        kama_up = kama_val > kama_prev
        kama_down = kama_val < kama_prev
        
        # Determine weekly KAMA direction for trend filter
        weekly_kama_up = weekly_kama_val > weekly_kama_prev
        weekly_kama_down = weekly_kama_val < weekly_kama_prev
        
        if position == 0:
            # LONG: KAMA turning up, RSI < 50, Chop > 61.8 (ranging market), and weekly KAMA up
            if kama_up and rsi_val < 50 and chop_val > 61.8 and weekly_kama_up:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down, RSI > 50, Chop > 61.8 (ranging market), and weekly KAMA down
            elif kama_down and rsi_val > 50 and chop_val > 61.8 and weekly_kama_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR Chop < 38.2 (trending market)
            if kama_down or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR Chop < 38.2 (trending market)
            if kama_up or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals