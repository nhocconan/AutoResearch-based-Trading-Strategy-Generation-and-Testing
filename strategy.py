#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily ATR filter and volume confirmation.
# In trending markets, price breaks beyond recent 20-period highs/lows with continuation.
# Uses daily ATR(14) to filter for volatility expansion and volume spike for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 20-50 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on daily data
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 13:
            atr_14_1d[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            if i == 13:
                atr_14_1d[i] = np.mean(tr[:14])
            else:
                atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels on 4h data (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 19:
            donchian_high[i] = np.max(high[:i+1]) if i > 0 else high[0]
            donchian_low[i] = np.min(low[:i+1]) if i > 0 else low[0]
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volatility expansion + volume spike
            if (close[i] > donchian_high[i] and 
                atr_14_1d_aligned[i] > atr_14_1d_aligned[i-1] * 1.2 and  # ATR increasing
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volatility expansion + volume spike
            elif (close[i] < donchian_low[i] and 
                  atr_14_1d_aligned[i] > atr_14_1d_aligned[i-1] * 1.2 and  # ATR increasing
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or volatility contraction
            if (close[i] < donchian_low[i] or 
                atr_14_1d_aligned[i] < atr_14_1d_aligned[i-1] * 0.8):  # ATR decreasing
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or volatility contraction
            if (close[i] > donchian_high[i] or 
                atr_14_1d_aligned[i] < atr_14_1d_aligned[i-1] * 0.8):  # ATR decreasing
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR14_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0