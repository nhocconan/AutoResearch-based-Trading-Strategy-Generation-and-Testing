#!/usr/bin/env python3
# [24910] 1d_1w_kama_rsi_chop_v1
# Hypothesis: Daily KAMA direction combined with RSI and weekly chop filter.
# Long when KAMA turns up, RSI < 70, and weekly chop > 61.8 (range) for mean reversion.
# Short when KAMA turns down, RSI > 30, and weekly chop > 61.8 (range) for mean reversion.
# Exit on opposite KAMA turn or RSI extreme reversal.
# Works in both bull and bear by using range-bound mean reversion during choppy markets.
# Target: 15-25 trades/year via strict KAMA turn + RSI + chop conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly chopiness index (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        n = len(close_arr)
        chop = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        # Calculate True Range
        tr = np.full(n, np.nan)
        for i in range(n):
            if i == 0:
                tr[i] = high_arr[i] - low_arr[i]
            else:
                tr[i] = max(high_arr[i] - low_arr[i], 
                           abs(high_arr[i] - close_arr[i-1]),
                           abs(low_arr[i] - close_arr[i-1]))
        
        # Calculate ATR
        for i in range(period-1, n):
            atr[i] = np.mean(tr[i-period+1:i+1])
        
        # Calculate Chop
        for i in range(period-1, n):
            if atr[i] > 0:
                highest_high = np.max(high_arr[i-period+1:i+1])
                lowest_low = np.min(low_arr[i-period+1:i+1])
                chop[i] = 100 * np.log10((highest_high - lowest_low) / (atr[i] * np.sqrt(period))) / np.log10(np.sqrt(period))
        return chop
    
    chop_1w = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate daily KAMA (10-period)
    def calculate_kama(close_arr, period=10):
        n = len(close_arr)
        kama = np.full(n, np.nan)
        if n < period:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close_arr, n=period))
        volatility = np.sum(np.abs(np.diff(close_arr)), axis=0) if len(close_arr) > 1 else 0
        er = np.zeros(n)
        er[:period] = np.nan
        for i in range(period, n):
            if volatility > 0:
                er[i] = change[i-period] / volatility
            else:
                er[i] = 0
        
        # Smoothing Constants
        sc = np.zeros(n)
        fast_sc = 2/(2+1)
        slow_sc = 2/(30+1)
        sc = er * (fast_sc - slow_sc) + slow_sc
        sc = sc * sc
        
        # KAMA
        kama[period-1] = np.mean(close_arr[:period])
        for i in range(period, n):
            kama[i] = kama[i-1] + sc[i] * (close_arr[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10)
    
    # Calculate daily RSI (14-period)
    def calculate_rsi(close_arr, period=14):
        n = len(close_arr)
        rsi = np.full(n, np.nan)
        if n < period + 1:
            return rsi
        
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        chop = chop_1w_aligned[i]
        price = close[i]
        kama_now = kama[i]
        kama_prev = kama[i-1] if i > 0 else kama_now
        rsi_now = rsi[i]
        
        kama_up = kama_now > kama_prev
        kama_down = kama_now < kama_prev
        
        if position == 1:  # Long
            # Exit: KAMA turns down or RSI > 70 (overbought)
            if kama_down or rsi_now > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: KAMA turns up or RSI < 30 (oversold)
            if kama_up or rsi_now < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: KAMA turns up, RSI < 70, and weekly chop > 61.8 (range)
            if kama_up and rsi_now < 70 and chop > 61.8:
                position = 1
                signals[i] = 0.25
            # Enter short: KAMA turns down, RSI > 30, and weekly chop > 61.8 (range)
            elif kama_down and rsi_now > 30 and chop > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals