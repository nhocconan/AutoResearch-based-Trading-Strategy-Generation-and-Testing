#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Choppiness Filter
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, filtered by RSI extremes
# and Choppiness Index to avoid ranging markets. Works in both bull and bear by
# following adaptive trend only when market is trending (CHOP < 38.2) or mean-reverting
# in extreme RSI during ranging markets (CHOP > 61.8). Target: 30-100 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1w data (higher timeframe for regime filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) on 1d ===
    def kama(arr, period=10, fast=2, slow=30):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        # Efficiency Ratio
        change = np.abs(np.diff(arr, period))
        volatility = np.sum(np.abs(np.diff(arr)), axis=0)
        er = np.zeros_like(arr)
        er[period:] = change[period-1:] / volatility[period-1:]
        er[er == 0] = 0
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_arr = np.zeros_like(arr)
        kama_arr[0] = arr[0]
        for i in range(1, len(arr)):
            kama_arr[i] = kama_arr[i-1] + sc[i] * (arr[i] - kama_arr[i-1])
        return kama_arr
    
    kama_1d = kama(close_1d, 10, 2, 30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === RSI(14) on 1d ===
    def rsi(arr, period=14):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_arr = 100 - (100 / (1 + rs))
        return rsi_arr
    
    rsi_1d = rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Choppiness Index on 1w ===
    def chop(high, low, close, period=14):
        if len(close) < period:
            return np.full_like(close, np.nan)
        atr = np.zeros_like(close)
        atr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        sum_atr = np.zeros_like(close)
        for i in range(period-1, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        chop_arr = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(period)
        return chop_arr
    
    chop_1w = chop(high_1w, low_1w, close_1w, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === 1d ATR for stoploss ===
    def atr(high, low, close, period=14):
        if len(close) < period:
            return np.full_like(close, np.nan)
        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_arr = np.zeros_like(close)
        for i in range(period-1, len(close)):
            atr_arr[i] = np.mean(tr[i-period+1:i+1])
        return atr_arr
    
    atr_1d = atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        chop_val = chop_1w_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price crosses below KAMA OR RSI > 70 in trending market OR RSI < 30 in ranging market
            if price < kama_val or (chop_val < 38.2 and rsi_val > 70) or (chop_val > 61.8 and rsi_val < 30):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above KAMA OR RSI < 30 in trending market OR RSI > 70 in ranging market
            if price > kama_val or (chop_val < 38.2 and rsi_val < 30) or (chop_val > 61.8 and rsi_val > 70):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trending market (CHOP < 38.2): follow KAMA direction with RSI filter
            if chop_val < 38.2:
                # Go long when price above KAMA and RSI > 50 (bullish momentum)
                if price > kama_val and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price below KAMA and RSI < 50 (bearish momentum)
                elif price < kama_val and rsi_val < 50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
            # Ranging market (CHOP > 61.8): mean reversion at RSI extremes
            elif chop_val > 61.8:
                # Go long when RSI < 30 (oversold) and price near low
                if rsi_val < 30 and price <= low[i] * 1.001:  # near daily low
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when RSI > 70 (overbought) and price near high
                elif rsi_val > 70 and price >= high[i] * 0.999:  # near daily high
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0