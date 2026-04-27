#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 4-hour Volume Weighted Average Price (VWAP) as dynamic support/resistance.
Price returning to VWAP after deviation shows mean-reversion tendency. Long when price > VWAP and rising,
short when price < VWAP and falling. Uses 12-hour EMA50 trend filter and volume confirmation (>1.5x avg).
Designed for low trade frequency (target 20-40/year) to minimize fee drag in ranging markets.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    close_12h = df_12h['close'].values
    ema_50 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_50[i] = (close_12h[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, np.nan)
    
    # 20-period average volume for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 50 for EMA50 seed, 20 for volume
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(vwap[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 12h EMA50
        bullish = ema_50_aligned[i] > ema_50_aligned[i-1]  # rising EMA
        bearish = ema_50_aligned[i] < ema_50_aligned[i-1]  # falling EMA
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price above VWAP, rising EMA trend, volume confirmation
            if price > vwap[i] and bullish and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: price below VWAP, falling EMA trend, volume confirmation
            elif price < vwap[i] and bearish and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below VWAP or trend turns bearish
            if price < vwap[i] or not bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above VWAP or trend turns bullish
            if price > vwap[i] or not bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_VWAP_EMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0