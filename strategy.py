#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation
Breakout from Camarilla R1/S1 levels with 1d trend filter and volume confirmation.
Long when price breaks above R1 with volume > 1.5x average and 1d EMA50 uptrend.
Short when price breaks below S1 with volume > 1.5x average and 1d EMA50 downtrend.
Exit when price returns to Pivot or trend fails.
Target: 20-40 trades/year per symbol.
"""

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
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    P = np.full(n, np.nan)
    
    for i in range(n):
        diff = prev_high[i] - prev_low[i]
        if diff > 0:
            R1[i] = prev_close[i] + 1.1 * diff / 12
            S1[i] = prev_close[i] - 1.1 * diff / 12
            P[i] = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_period + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    start_idx = max(1, ema_1d_period - 1, vol_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(P[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        ema1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume confirmation and 1d uptrend
            if (price > R1[i] and vol_ratio > 1.5 and price > ema1d_val):
                signals[i] = size
                position = 1
            # Short: break below S1 with volume confirmation and 1d downtrend
            elif (price < S1[i] and vol_ratio > 1.5 and price < ema1d_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to Pivot or trend fails
            if (price <= P[i] or price < ema1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: return to Pivot or trend fails
            if (price >= P[i] or price > ema1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0