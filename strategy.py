#!/usr/bin/env python3
# [24941] 4h_1d_rsi_pullback_trend_v1
# Hypothesis: 4-hour RSI pullback in direction of 1-day trend. Long when RSI < 30 and price above 1-day EMA200; short when RSI > 70 and price below 1-day EMA200. Exit when RSI returns to neutral (40-60). Uses volume confirmation to avoid false signals. Designed to work in both bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend) markets with controlled trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_pullback_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend
    close_1d = df_1d['close'].values
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2/201) + (ema200_1d[i-1] * (1 - 2/201))
    
    # Calculate RSI(14) on 4h
    rsi = np.full(n, np.nan)
    if n >= 15:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[14] = np.mean(gain[:14])
        avg_loss[14] = np.mean(loss[:14])
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day EMA200 to 4h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        rsi_val = rsi[i]
        ema200 = ema200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: RSI returns to neutral zone (40-60)
            if rsi_val >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI returns to neutral zone (40-60)
            if rsi_val <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI oversold (<30), price above 1-day EMA200 (uptrend), volume confirmation
            if rsi_val < 30 and price > ema200 and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI overbought (>70), price below 1-day EMA200 (downtrend), volume confirmation
            elif rsi_val > 70 and price < ema200 and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals