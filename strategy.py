#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Uses daily EMA34 for trend direction, 12h Donchian(20) channels for entry signals,
# and volume confirmation to filter false breakouts. Works in both bull and bear markets
# by following the higher timeframe trend while entering on lower timeframe breakouts.
# Target: 25-35 trades/year to minimize fee decay while capturing trend continuation moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Get 12h data for Donchian channels and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d for trend
    close_1d = df_1d['close'].values
    ema_len = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate 20-period Donchian channels on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    dc_len = 20
    dc_upper_12h = np.full(len(high_12h), np.nan)
    dc_lower_12h = np.full(len(low_12h), np.nan)
    
    for i in range(dc_len - 1, len(high_12h)):
        dc_upper_12h[i] = np.max(high_12h[i-dc_len+1:i+1])
        dc_lower_12h[i] = np.min(low_12h[i-dc_len+1:i+1])
    
    # Calculate 20-period average volume on 12h for spike detection
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    vol_period = 20
    for i in range(vol_period, len(vol_12h)):
        vol_ma_12h[i] = np.mean(vol_12h[i-vol_period:i])
    
    # Align all indicators to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    dc_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, dc_upper_12h)
    dc_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, dc_lower_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(35, 25) + 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(dc_upper_12h_aligned[i]) or 
            np.isnan(dc_lower_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_12h_aligned[i] if vol_ma_12h_aligned[i] > 0 else 0
        
        # Volume confirmation: at least 1.8x average volume
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: Donchian breakout above upper channel with uptrend and volume
            if price > dc_upper_12h_aligned[i] and price > ema_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Donchian breakdown below lower channel with downtrend and volume
            elif price < dc_lower_12h_aligned[i] and price < ema_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian lower channel or trend reversal
            if price < dc_lower_12h_aligned[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian upper channel or trend reversal
            if price > dc_upper_12h_aligned[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0