# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot reversal with 1d EMA50 trend filter and volume confirmation.
Go long when price touches S1/S2 in bullish trend, short when touches R1/R2 in bearish trend.
Volume > 2x average confirms reversal strength. Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.
Target: 20-40 trades/year (80-160 over 4 years). Includes ATR-based stoploss to limit drawdown.
Works in both bull and bear markets by trading reversals from institutional pivot levels.
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
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_S1 = np.full(len(close_1d), np.nan)
    camarilla_S2 = np.full(len(close_1d), np.nan)
    camarilla_R1 = np.full(len(close_1d), np.nan)
    camarilla_R2 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        rng = high_1d[i] - low_1d[i]
        camarilla_S1[i] = close_1d[i] - 1.0486 * rng / 6
        camarilla_S2[i] = close_1d[i] - 1.0486 * rng / 4
        camarilla_R1[i] = close_1d[i] + 1.0486 * rng / 6
        camarilla_R2[i] = close_1d[i] + 1.0486 * rng / 4
    
    # Align Camarilla levels to 12h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R2)
    
    # ATR for stoploss
    atr_period = 14
    tr = np.zeros(n)
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:atr_period+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Volume average for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 50 for EMA50, 20 for volume
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA50
        bullish = price > ema_50_aligned[i]
        bearish = price < ema_50_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long reversal: price touches S1/S2 in bullish trend with volume
            if bullish and volume_confirmation and (price <= S1_aligned[i] or price <= S2_aligned[i]):
                signals[i] = size
                position = 1
            # Short reversal: price touches R1/R2 in bearish trend with volume
            elif bearish and volume_confirmation and (price >= R1_aligned[i] or price >= R2_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches opposite pivot (R1) or trend turns bearish or stoploss hit
            if price >= R1_aligned[i] or bearish or price < (entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price reaches opposite pivot (S1) or trend turns bullish or stoploss hit
            if price <= S1_aligned[i] or bullish or price > (entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
        
        # Track entry price for stoploss calculation
        if position != 0 and signals[i] != 0:
            if position == 1 and signals[i] == size:
                entry_price = price
            elif position == -1 and signals[i] == -size:
                entry_price = price
    
    return signals

name = "12h_Camarilla_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0