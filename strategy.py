#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot reversal strategy with volume confirmation and 1d trend filter.
Uses Camarilla levels calculated from previous 1d high/low/close. Long at S1 in bullish trend,
short at R1 in bearish trend. Volume > 2x average confirms reversal strength. Designed for
mean reversion in ranging markets and breakout alignment in trending markets.
Target: 30-60 trades/year (120-240 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate previous day's Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla multipliers
    # R4 = close + (high-low) * 1.1/2
    # R3 = close + (high-low) * 1.1/4
    # R2 = close + (high-low) * 1.1/6
    # R1 = close + (high-low) * 1.1/12
    # S1 = close - (high-low) * 1.1/12
    # S2 = close - (high-low) * 1.1/6
    # S3 = close - (high-low) * 1.1/4
    # S4 = close - (high-low) * 1.1/2
    
    camarilla_range = (prev_high - prev_low) * 1.1
    r1 = prev_close + camarilla_range / 12
    s1 = prev_close - camarilla_range / 12
    
    # Align Camarilla levels to 4h (they're based on previous day, so available at 00:00 UTC)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 20-period average volume for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 1 for Camarilla (previous day), 20 for volume, 50 for EMA50
    start_idx = max(1, vol_period, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
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
            # Long at S1 in bullish trend with volume confirmation
            if bullish and price <= s1_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short at R1 in bearish trend with volume confirmation
            elif bearish and price >= r1_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches R1 or trend turns bearish
            if price >= r1_aligned[i] or bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price reaches S1 or trend turns bullish
            if price <= s1_aligned[i] or bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0