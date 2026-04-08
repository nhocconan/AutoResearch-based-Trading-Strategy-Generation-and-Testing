#!/usr/bin/env python3
# [24929] 4h_1d_atr_breakout_v1
# Hypothesis: 4-hour ATR-based breakout with 1-day trend filter and volume confirmation.
# Long when price closes above previous close + 1.5*ATR(14) with volume > 1.8x average and price > 1-day EMA50.
# Short when price closes below previous close - 1.5*ATR(14) with volume > 1.8x average and price < 1-day EMA50.
# Exit when price reverses by 1.0*ATR(14) or volume falls below 1.3x average.
# Designed to capture momentum bursts in both bull and bear markets with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_atr_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Calculate ATR(14)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n >= 14:
        atr[13] = np.mean(tr[1:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day EMA50 to 4-hour timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        prev_close = close[i-1]
        atr_val = atr[i]
        
        if position == 1:  # Long
            # Exit: price reverses by 1.0*ATR or volume drops below 1.3x average
            if price < prev_close - 1.0 * atr_val or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reverses by 1.0*ATR or volume drops below 1.3x average
            if price > prev_close + 1.0 * atr_val or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above previous close + 1.5*ATR with volume expansion and uptrend on 1d
            if price > prev_close + 1.5 * atr_val and vol_ratio > 1.8 and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below previous close - 1.5*ATR with volume expansion and downtrend on 1d
            elif price < prev_close - 1.5 * atr_val and vol_ratio > 1.8 and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals