#!/usr/bin/env python3
"""
6h Ehlers Fisher Transform with Volume Spike and 12h Trend Filter
Hypothesis: The Fisher Transform identifies extreme price reversals with high accuracy.
Combined with volume confirmation and higher-timeframe trend alignment, it provides
reliable entry points in both bull and bear markets by fading extremes in the trend direction.
Works well on 6h timeframe with lower trade frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def fisher_transform(price, length=10):
    """Ehlers Fisher Transform"""
    if len(price) < length:
        return np.full_like(price, np.nan), np.full_like(price, np.nan)
    
    # Normalize price to [-1, 1] range
    highest = np.full(len(price), np.nan)
    lowest = np.full(len(price), np.nan)
    
    for i in range(length-1, len(price)):
        highest[i] = np.max(price[i-length+1:i+1])
        lowest[i] = np.min(price[i-length+1:i+1])
    
    # Avoid division by zero
    diff = highest - lowest
    diff = np.where(diff == 0, 1e-10, diff)
    
    value1 = np.where(diff != 0, 2 * ((price - lowest) / diff - 0.5), 0)
    value1 = np.clip(value1, -0.999, 0.999)
    
    # Apply smoothing and Fisher transform
    value2 = np.full_like(price, np.nan)
    for i in range(1, len(price)):
        if np.isnan(value1[i]):
            value2[i] = value2[i-1] if not np.isnan(value2[i-1]) else 0
        else:
            value2[i] = 0.33 * value1[i] + 0.67 * (value2[i-1] if not np.isnan(value2[i-1]) else 0)
    
    fish = np.full_like(price, np.nan)
    for i in range(1, len(price)):
        if np.isnan(value2[i]):
            fish[i] = fish[i-1] if not np.isnan(fish[i-1]) else 0
        else:
            fish[i] = 0.5 * np.log((1 + value2[i]) / (1 - value2[i]) + 0.1) + 0.5 * (fish[i-1] if not np.isnan(fish[i-1]) else 0)
    
    return fish, value2  # fish, trigger

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA on 12h close for trend direction
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    multiplier = 2 / (34 + 1)  # EMA 34
    for i in range(len(close_12h)):
        if i == 0:
            ema_12h[i] = close_12h[i]
        elif not np.isnan(close_12h[i]):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
        else:
            ema_12h[i] = ema_12h[i-1]
    
    # Align 12h EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Fisher Transform on 6h price (typical price)
    typical_price = (high + low + close) / 3
    fish, trigger = fisher_transform(typical_price, length=10)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 20:
            if i >= 0:
                vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(fish[i]) or np.isnan(trigger[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long signal: Fisher crosses above trigger AND below -1.5 (oversold) AND price above 12h EMA (uptrend)
            if (fish[i] > trigger[i] and fish[i-1] <= trigger[i-1] and 
                fish[i] < -1.5 and close[i] > ema_12h_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short signal: Fisher crosses below trigger AND above +1.5 (overbought) AND price below 12h EMA (downtrend)
            elif (fish[i] < trigger[i] and fish[i-1] >= trigger[i-1] and 
                  fish[i] > 1.5 and close[i] < ema_12h_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below trigger OR price crosses below 12h EMA
            if (fish[i] < trigger[i] and fish[i-1] >= trigger[i-1]) or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fisher crosses above trigger OR price crosses above 12h EMA
            if (fish[i] > trigger[i] and fish[i-1] <= trigger[i-1]) or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_FisherTransform_VolumeSpike_12hEMAFilter"
timeframe = "6h"
leverage = 1.0