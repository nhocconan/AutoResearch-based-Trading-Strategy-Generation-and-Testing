#!/usr/bin/env python3
"""
12h strategy using 1-day KAMA trend with RSI mean-reversion and 1-week ADX filter.
Long when KAMA turns up, RSI < 35, and ADX > 20 (trending or ranging).
Short when KAMA turns down, RSI > 65, and ADX > 20.
Exit when RSI crosses 50 (mean reversion) or ADX < 15 (no trend).
Designed for low turnover: ~15-25 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data once for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (10, 2, 30)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, kama_period))
    abs_change = np.abs(np.diff(close_1d))
    er = np.zeros(len(close_1d))
    er[kama_period:] = change[kama_period:] / np.sum(np.lib.stride_tricks.sliding_window_view(abs_change, kama_period), axis=1)
    er = np.where(np.isnan(er), 0, er)
    
    # Smoothing Constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    sc = np.where(np.isnan(sc), 0, sc)
    
    # KAMA
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14)
    rsi_period = 14
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(close_1d))
    avg_loss = np.zeros(len(close_1d))
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period+1, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14)
    adx_period = 14
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.zeros(len(close_1w))
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (adx_period-1) + tr[i]) / adx_period
    
    # Directional Movement
    up = high_1w - np.roll(high_1w, 1)
    down = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed DM
    plus_dm_smooth = np.zeros(len(close_1w))
    minus_dm_smooth = np.zeros(len(close_1w))
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    for i in range(1, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (adx_period-1) + plus_dm[i]) / adx_period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (adx_period-1) + minus_dm[i]) / adx_period
    
    # Directional Indicators
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros(len(close_1w))
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 1-day indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Align 1-week ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(adx_aligned[i]):
            continue
        
        if position == 0:
            # Long: KAMA turning up, RSI < 35 (oversold), ADX > 20
            if i > 0 and kama_aligned[i] > kama_aligned[i-1] and rsi_aligned[i] < 35 and adx_aligned[i] > 20:
                position = 1
                signals[i] = position_size
            # Short: KAMA turning down, RSI > 65 (overbought), ADX > 20
            elif i > 0 and kama_aligned[i] < kama_aligned[i-1] and rsi_aligned[i] > 65 and adx_aligned[i] > 20:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: RSI crosses 50 (mean reversion) or ADX < 15 (no trend)
            if rsi_aligned[i] >= 50 or adx_aligned[i] < 15:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: RSI crosses 50 (mean reversion) or ADX < 15 (no trend)
            if rsi_aligned[i] <= 50 or adx_aligned[i] < 15:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_KAMA_RSI_1wADX"
timeframe = "12h"
leverage = 1.0