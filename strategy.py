#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_TrendFilter
# Hypothesis: Breakout above/below Donchian(20) channel with volume >1.5x 20-bar average and trend filter from 1d EMA50.
# Works in bull markets via upside breakouts and in bear via downside breakdowns. Volume confirms momentum, EMA50 filters counter-trend noise.
# Target: 20-40 trades/year on 4h timeframe.

name = "4h_Donchian_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) with proper initialization
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period high/low)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    if len(high) >= 20:
        # Initialize first value
        donchian_high[19] = np.max(high[0:20])
        donchian_low[19] = np.min(low[0:20])
        # Rolling calculation
        for i in range(20, len(high)):
            donchian_high[i] = max(donchian_high[i-1], high[i])
            donchian_low[i] = min(donchian_low[i-1], low[i])
            # Remove oldest value from window
            if i >= 40:
                if donchian_high[i] == high[i-20]:
                    donchian_high[i] = np.max(high[i-19:i+1])
                if donchian_low[i] == low[i-20]:
                    donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: 4h volume / 20-period average volume
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
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or \
           np.isnan(donchian_low[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above Donchian high AND volume confirmation AND bullish trend (price > EMA50)
            if close[i] > donchian_high[i] and volume_ratio[i] > 1.5 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian low AND volume confirmation AND bearish trend (price < EMA50)
            elif close[i] < donchian_low[i] and volume_ratio[i] > 1.5 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Donchian low (reversal signal) or trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Donchian high (reversal signal) or trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals