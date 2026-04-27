#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses daily EMA50 for trend direction, 4h Donchian channels for entry signals,
# and volume > 1.5x average to filter false breakouts. Works in both bull and bear markets
# by following higher timeframe trend while entering on 4h breakouts.
# Target: 20-35 trades/year to minimize fee decay while capturing trend continuation moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d for trend
    close_1d = df_1d['close'].values
    ema_len = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate 20-period Donchian channels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    dc_len = 20
    dc_upper_4h = np.full(len(high_4h), np.nan)
    dc_lower_4h = np.full(len(low_4h), np.nan)
    
    for i in range(dc_len - 1, len(high_4h)):
        dc_upper_4h[i] = np.max(high_4h[i-dc_len+1:i+1])
        dc_lower_4h[i] = np.min(low_4h[i-dc_len+1:i+1])
    
    # Calculate 20-period average volume on 4h for spike detection
    vol_4h = df_4h['volume'].values
    vol_ma_4h = np.full(len(vol_4h), np.nan)
    vol_period = 20
    for i in range(vol_period, len(vol_4h)):
        vol_ma_4h[i] = np.mean(vol_4h[i-vol_period:i])
    
    # Align all indicators to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    dc_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_upper_4h)
    dc_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_lower_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, 25) + 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(dc_upper_4h_aligned[i]) or 
            np.isnan(dc_lower_4h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_4h_aligned[i] if vol_ma_4h_aligned[i] > 0 else 0
        
        # Volume confirmation: at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: Donchian breakout above upper channel with uptrend and volume
            if price > dc_upper_4h_aligned[i] and price > ema_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Donchian breakdown below lower channel with downtrend and volume
            elif price < dc_lower_4h_aligned[i] and price < ema_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian lower channel or trend reversal
            if price < dc_lower_4h_aligned[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian upper channel or trend reversal
            if price > dc_upper_4h_aligned[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0