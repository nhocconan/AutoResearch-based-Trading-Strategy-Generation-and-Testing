#!/usr/bin/env python3
"""
12h_DonchianBreakout_1dTrend_Volume
Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation.
Long when price breaks above 20-period high in 1d uptrend with volume > 1.5x average.
Short when price breaks below 20-period low in 1d downtrend with volume > 1.5x average.
Exit when price touches the opposite Donchian band or trend reverses.
Designed to capture trends while minimizing whipsaw in choppy markets.
Target: 20-30 trades/year to stay well below fee drag thresholds.
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
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    
    for i in range(19, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-19:i+1])
        donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate 12h volume moving average (20-period)
    volume_12h = df_12h['volume'].values
    vol_ma_period = 20
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(vol_ma_period, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-vol_ma_period:i+1])
    
    # Align all 12h and 1d indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 50, 20)  # Donchian(20), EMA(50), vol MA(20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average 12h volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend with volume
            if uptrend and volume_confirmation and price > donchian_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in downtrend with volume
            elif downtrend and volume_confirmation and price < donchian_low_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price touches Donchian low or trend reverses
            if price <= donchian_low_aligned[i] or price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price touches Donchian high or trend reverses
            if price >= donchian_high_aligned[i] or price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_DonchianBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0