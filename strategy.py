#!/usr/bin/env python3
"""
6h_RSI_Trend_Pullback
Trend-following pullback strategy using RSI(14) and EMA(50) on 6h timeframe.
Long when RSI < 40 (pullback) and price > EMA50 (uptrend).
Short when RSI > 60 (pullback) and price < EMA50 (downtrend).
Uses 1d ADX(14) > 25 to confirm strong trend and avoid range-bound whipsaws.
Exit when RSI returns to neutral (40-60) or trend filter fails.
Target: 20-40 trades/year per symbol.
"""

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
    
    # RSI(14) calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[rsi_period:i+1])
            avg_loss[i] = np.mean(loss[rsi_period:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rsi = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # EMA50 for trend filter
    ema_period = 50
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period-1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema[i] = (close[i] * (2 / (ema_period + 1)) + 
                      ema[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Get 1d data for ADX trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) calculation
    adx_period = 14
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM
    atr = np.full(len(tr), np.nan)
    plus_di = np.full(len(tr), np.nan)
    minus_di = np.full(len(tr), np.nan)
    
    for i in range(adx_period, len(tr)):
        if i == adx_period:
            atr[i] = np.sum(tr[adx_period:i+1])
            plus_dm_sum = np.sum(plus_dm[adx_period:i+1])
            minus_dm_sum = np.sum(minus_dm[adx_period:i+1])
        else:
            atr[i] = atr[i-1] - (atr[i-1] / adx_period) + tr[i]
            plus_dm_sum = plus_di[i-1] * (adx_period-1) + plus_dm[i]
            minus_dm_sum = minus_di[i-1] * (adx_period-1) + minus_dm[i]
        
        if atr[i] != 0:
            plus_di[i] = 100 * (plus_dm_sum / atr[i])
            minus_di[i] = 100 * (minus_dm_sum / atr[i])
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # Calculate DX and ADX
    dx = np.full(len(tr), np.nan)
    adx = np.full(len(tr), np.nan)
    
    for i in range(adx_period, len(tr)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    for i in range(2*adx_period-1, len(tr)):
        if i == 2*adx_period-1:
            adx[i] = np.mean(dx[adx_period:i+1])
        else:
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI, EMA50, and ADX
    start_idx = max(rsi_period, ema_period, 2*adx_period-1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_val = ema[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: RSI < 40 (pullback) and price > EMA50 (uptrend) and ADX > 25 (strong trend)
            if (rsi_val < 40 and price > ema_val and adx_val > 25):
                signals[i] = size
                position = 1
            # Short: RSI > 60 (pullback) and price < EMA50 (downtrend) and ADX > 25 (strong trend)
            elif (rsi_val > 60 and price < ema_val and adx_val > 25):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or trend fails
            if (rsi_val >= 40 and rsi_val <= 60) or price < ema_val or adx_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or trend fails
            if (rsi_val >= 40 and rsi_val <= 60) or price > ema_val or adx_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_Trend_Pullback"
timeframe = "6h"
leverage = 1.0