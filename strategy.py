#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing reliable trend direction.
In ranging markets (high Chop), we avoid trades; in trending markets (low Chop), we follow KAMA direction with RSI filter for entry timing.
Works in both bull and bear markets by adapting to volatility and using Chop to filter regimes.
Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # --- 1d data for Chop and RSI ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Chop calculation
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Chop: EMA of TR / (max(high) - min(low)) over 14 days
    atr_14 = np.zeros(len(tr1))
    for i in range(len(tr1)):
        if i < 13:
            atr_14[i] = np.mean(tr1[:i+1]) if i >= 0 else tr1[i]
        else:
            atr_14[i] = np.mean(tr1[i-13:i+1])
    
    max_high_14 = np.zeros(len(high_1d))
    min_low_14 = np.zeros(len(low_1d))
    for i in range(len(high_1d)):
        if i < 13:
            max_high_14[i] = np.max(high_1d[:i+1])
            min_low_14[i] = np.min(low_1d[:i+1])
        else:
            max_high_14[i] = np.max(high_1d[i-13:i+1])
            min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    range_14[range_14 == 0] = 1e-10
    
    chop_raw = (np.sum(atr_14[-14:]) / range_14[-1]) * 100 if len(atr_14) >= 14 else 50
    # For simplicity, we use a rolling Chop approximation; in practice, precompute full series
    chop = np.full(len(high_1d), 50.0)  # placeholder; will compute properly below
    
    # Proper Chop calculation over rolling window
    chop = np.full(len(high_1d), np.nan)
    for i in range(13, len(high_1d)):
        atr_sum = np.sum(atr_14[i-13:i+1])
        r = max_high_14[i] - min_low_14[i]
        if r > 0:
            chop[i] = (atr_sum / r) * 100
        else:
            chop[i] = 50.0
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 13:
            avg_gain[i] = np.mean(gain[:i+1]) if i >= 0 else gain[i]
            avg_loss[i] = np.mean(loss[:i+1]) if i >= 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # --- 4h data for KAMA ---
    close = prices['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=0)), axis=0) if hasattr(np.sum, 'axis') else None
    # Manual volatility sum over 10 periods
    volatility = np.zeros_like(close)
    for i in range(len(close)):
        if i < 10:
            volatility[i] = np.sum(np.abs(np.diff(close[:i+1], prepend=0)))
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1], prepend=0)))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- Volume filter: 1.5x 20-period average ---
    volume = prices['volume'].values
    vol_avg = np.zeros(n)
    for i in range(n):
        if i < 19:
            vol_avg[i] = np.mean(volume[:i+1]) if i >= 0 else volume[i]
        else:
            vol_avg[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # warmup for indicators
        if np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        chop_val = chop_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_confirm = volume_filter[i]
        
        # Chop filter: only trade when Chop < 50 (trending market)
        if chop_val >= 50:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA and RSI > 50 (bullish momentum)
            if price > kama_val and rsi_val > 50 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA and RSI < 50 (bearish momentum)
            elif price < kama_val and rsi_val < 50 and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < KAMA or RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA or RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0