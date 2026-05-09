#!/usr/bin/env python3
# 1d_KAMA_Trend_Filter
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) as a trend filter with RSI and chop filter on daily timeframe.
# Works in bull/bear: KAMA adapts to market noise, reducing false signals in ranging markets.
# RSI filters overbought/oversold conditions, chop filter ensures trades only in trending regimes.
# Uses weekly trend for higher timeframe confirmation to avoid counter-trend trades.

name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
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
    
    # Calculate KAMA (10, 2, 30) for trend
    def kama(close, fast=2, slow=30, length=10):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / np.where(volatility[length-1:] == 0, 1, volatility[length-1:])
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[length-1] = np.mean(close[:length])
        for i in range(length, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close)
    
    # Calculate RSI (14)
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_vals = rsi(close)
    
    # Calculate Choppiness Index (14)
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        # Wilder's smoothing
        atr_len = np.zeros_like(close)
        atr_len[length] = np.sum(atr[1:length+1])
        for i in range(length+1, len(close)):
            atr_len[i] = atr_len[i-1] - (atr_len[i-1] / length) + atr[i]
        # Sum of true range over period
        sum_tr = atr_len
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(length, len(close)):
            max_high[i] = np.max(high[i-length+1:i+1])
            min_low[i] = np.min(low[i-length+1:i+1])
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(length, len(close)):
            if max_high[i] - min_low[i] != 0:
                chop[i] = 100 * np.log10(sum_tr[i] / (max_high[i] - min_low[i])) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    chop_vals = chop(high, low, close)
    
    # Get weekly trend (EWMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Ensure KAMA, RSI, CHOP are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_vals[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > KAMA (uptrend), RSI < 60 (not overbought), CHOP < 61.8 (trending), weekly uptrend
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] < 60 and 
                chop_vals[i] < 61.8 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA (downtrend), RSI > 40 (not oversold), CHOP < 61.8 (trending), weekly downtrend
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] > 40 and 
                  chop_vals[i] < 61.8 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA OR RSI > 70 (overbought) OR CHOP > 61.8 (ranging)
            if (close[i] < kama_vals[i] or 
                rsi_vals[i] > 70 or 
                chop_vals[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA OR RSI < 30 (oversold) OR CHOP > 61.8 (ranging)
            if (close[i] > kama_vals[i] or 
                rsi_vals[i] < 30 or 
                chop_vals[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals