#!/usr/bin/env python3
# 12h_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in sideways markets. Combined with RSI extremes and 1d trend filter, it captures strong momentum moves while avoiding false signals in chop. Works in bull/bear: trend filter ensures trades align with higher timeframe direction, RSI avoids overextended entries.

name = "12h_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate KAMA on 12h
    def kama(price, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.sum(np.abs(np.diff(price)), axis=0) if len(price) == 1 else np.convolve(np.abs(np.diff(price)), np.ones(er_length), 'same')
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.full_like(price, np.nan)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def rsi(price, length=14):
        delta = np.diff(price, prepend=price[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(length)/length, 'full')[:len(price)]
        avg_loss = np.convolve(loss, np.ones(length)/length, 'full')[:len(price)]
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate KAMA and RSI on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    kama_12h = kama(close_12h, er_length=10, fast_sc=2, slow_sc=30)
    rsi_12h = rsi(close_12h, length=14)
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume filter: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > KAMA AND RSI < 30 (oversold) AND uptrend (price > 1d EMA50) AND volume spike
            if (close[i] > kama_12h_aligned[i] and 
                rsi_12h_aligned[i] < 30 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA AND RSI > 70 (overbought) AND downtrend (price < 1d EMA50) AND volume spike
            elif (close[i] < kama_12h_aligned[i] and 
                  rsi_12h_aligned[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA OR RSI > 70 (overbought) OR trend change
            if (close[i] < kama_12h_aligned[i] or 
                rsi_12h_aligned[i] > 70 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA OR RSI < 30 (oversold) OR trend change
            if (close[i] > kama_12h_aligned[i] or 
                rsi_12h_aligned[i] < 30 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals