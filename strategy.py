# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d EMA trend filter and volume confirmation
# Uses 1d EMA50 for trend direction, 12h Donchian(20) breakouts for entry signals,
# and volume spikes (1.5x 20-period average) to confirm breakouts. Works in both bull and bear
# markets by following the 1d trend while entering on Donchian breakouts. Target: 20-30 trades/year
# to minimize fee decay while capturing trend continuation moves. Focus on BTC/ETH with ETH as primary.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
    
    # Get 12h data for Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    n_12h = len(high_12h)
    
    # Calculate 20-period Donchian channels on 12h
    donchian_period = 20
    upper_12h = np.full(n_12h, np.nan)
    lower_12h = np.full(n_12h, np.nan)
    
    for i in range(donchian_period - 1, n_12h):
        upper_12h[i] = np.max(high_12h[i-donchian_period+1:i+1])
        lower_12h[i] = np.min(low_12h[i-donchian_period+1:i+1])
    
    # Align HTF indicators to LTF
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Calculate 20-period average volume on 12h for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, 20) + 20  # EMA50 needs 50, Donchian needs 20, vol needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: Donchian upper breakout with uptrend and volume
            if price > upper_12h_aligned[i] and price > ema_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Donchian lower breakdown with downtrend and volume
            elif price < lower_12h_aligned[i] and price < ema_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian lower or trend reversal
            if price < lower_12h_aligned[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian upper or trend reversal
            if price > upper_12h_aligned[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0