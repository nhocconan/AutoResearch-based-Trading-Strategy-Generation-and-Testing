#!/usr/bin/env python3
"""
4h_Donchian20_1dVolumeSpike_TrendFilter
Strategy: 4h Donchian breakout with 1d volume spike and trend filter.
Long: Close > upper band + 1d volume > 1.5x 20-day avg + 1d close > 1d open
Short: Close < lower band + 1d volume > 1.5x 20-day avg + 1d close < 1d open
Exit: Opposite band touch
Position size: 0.25
Designed to capture strong momentum moves with volume confirmation and trend alignment.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian channel (20-period)
    upper = np.full_like(close, np.nan)
    lower = np.full_like(close, np.nan)
    
    # Calculate rolling max/min
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume spike (>1.5x 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma20 = np.full_like(vol_1d, np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma20[i] = np.mean(vol_1d[i-20:i])
    vol_spike = np.where(vol_ma20 > 0, vol_1d / vol_ma20, 0)
    vol_spike_filter = align_htf_to_ltf(prices, df_1d, vol_spike > 1.5)
    
    # 1d trend (close > open = uptrend)
    trend_1d = (df_1d['close'] > df_1d['open']).astype(float).values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Start from sufficient warmup
    start_idx = 40  # 20 for Donchian + buffer
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_spike_filter[i])):
            signals[i] = 0.0
            continue
        
        # Entry signals
        if position == 0:
            # Long: break above upper + volume spike + 1d uptrend
            if close[i] > upper[i] and vol_spike_filter[i] and trend_1d_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: break below lower + volume spike + 1d downtrend
            elif close[i] < lower[i] and vol_spike_filter[i] and trend_1d_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch or cross lower band
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch or cross upper band
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0