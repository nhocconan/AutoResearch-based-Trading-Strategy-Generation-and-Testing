#!/usr/bin/env python3
# 12h_PriceChannel_Breakout_1dTrend_Volume
# Hypothesis: Breakout above/below 12h Donchian(20) channels with 1d trend filter (EMA50) and volume confirmation (>1.5x 20-bar avg).
# Works in bull/bear markets by following the 1d trend while using 12h price channels for entry.
# Target: 15-25 trades/year on 12h timeframe, focusing on BTC/ETH pairs.

name = "12h_PriceChannel_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = np.full_like(high_12h, np.nan)
    low_20 = np.full_like(low_12h, np.nan)
    for i in range(len(high_12h)):
        if i >= 19:
            high_20[i] = np.max(high_12h[i-19:i+1])
            low_20[i] = np.min(low_12h[i-19:i+1])
    
    # Align 12h Donchian to 12h timeframe (same as input)
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 12h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above 12h Donchian high AND bullish 1d trend AND volume confirmation
            if close[i] > high_20_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below 12h Donchian low AND bearish 1d trend AND volume confirmation
            elif close[i] < low_20_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below 12h Donchian low or trend turns bearish
            if close[i] < low_20_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above 12h Donchian high or trend turns bullish
            if close[i] > high_20_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals