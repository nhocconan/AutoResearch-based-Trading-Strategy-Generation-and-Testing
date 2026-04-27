#!/usr/bin/env python3
"""
4h_MedianTrend_12hATR_Exit
Hypothesis: Price closes above/below 4h median price (HL2) with volume confirmation and 12h trend filter.
Long when close > HL2, volume > 1.3x average, and 12h uptrend.
Short when close < HL2, volume > 1.3x average, and 12h downtrend.
Exit when trend reverses or volume drops.
Uses median price to reduce noise from outliers. Designed for 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_12h_period = 34
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_12h_period:
        ema_12h[ema_12h_period - 1] = np.mean(close_12h[:ema_12h_period])
        multiplier = 2 / (ema_12h_period + 1)
        for i in range(ema_12h_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Calculate 4h median price (HL2)
    hl2 = (high + low) / 2
    
    # Calculate 4h volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 34)  # volume MA needs 20, EMA needs 34
    
    for i in range(start_idx, n):
        if (np.isnan(hl2[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Trend filter: 12h EMA34
        uptrend = price > ema_12h_aligned[i]
        downtrend = price < ema_12h_aligned[i]
        
        # Volume confirmation: > 1.3x average volume
        volume_confirmation = vol_ratio > 1.3
        
        if position == 0:
            # Long: close above median with volume and uptrend
            if uptrend and volume_confirmation and price > hl2[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below median with volume and downtrend
            elif downtrend and volume_confirmation and price < hl2[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: trend reversal or volume drop
            if price <= hl2[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: trend reversal or volume drop
            if price >= hl2[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_MedianTrend_12hATR_Exit"
timeframe = "4h"
leverage = 1.0