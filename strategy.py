#!/usr/bin/env python3
# [24974] 1h_4h_1d_rsi_momentum_v1
# Hypothesis: 1-hour RSI momentum with 4-hour trend filter (price > EMA50) and 1-day volume confirmation.
# Long when RSI(14) > 55, price > 4h EMA50, and 1-day volume > 1.5x average.
# Short when RSI(14) < 45, price < 4h EMA50, and 1-day volume > 1.5x average.
# Exit when RSI returns to neutral zone (45-55).
# Uses 4-hour EMA50 for trend bias and 1-day volume for conviction, effective in both trending and ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4-hour data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4-hour EMA50
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema = np.zeros(len(close_4h))
        ema[0] = close_4h[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_4h)):
            ema[i] = alpha * close_4h[i] + (1 - alpha) * ema[i-1]
        ema50_4h[49:] = ema[49:]
    
    # Get 1-day data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1-day volume moving average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    if len(vol_1d) >= 20:
        for i in range(20, len(vol_1d)):
            vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Calculate RSI(14)
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
    
    # Align 4-hour EMA50 to 1-hour timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Align 1-day volume MA to 1-hour timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        rsi_val = rsi[i]
        price = close[i]
        
        if position == 1:  # Long
            # Exit: RSI returns to neutral zone (<= 55)
            if rsi_val <= 55:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: RSI returns to neutral zone (>= 45)
            if rsi_val >= 45:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: RSI > 55, price > 4h EMA50, and volume > 1.5x average
            if rsi_val > 55 and price > ema50_4h_aligned[i] and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.20
            # Enter short: RSI < 45, price < 4h EMA50, and volume > 1.5x average
            elif rsi_val < 45 and price < ema50_4h_aligned[i] and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.20
    
    return signals