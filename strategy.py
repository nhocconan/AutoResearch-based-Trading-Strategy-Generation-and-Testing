#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + EMA Trend Filter
Hypothesis: Donchian breakouts capture strong trending moves. Volume spikes confirm institutional participation.
EMA filter ensures alignment with higher timeframe trend. Works in both bull and bear markets by catching
breakouts in either direction. Low trade frequency due to strict breakout conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    donchian_high = np.zeros_like(high)
    donchian_low = np.zeros_like(low)
    
    for i in range(len(high)):
        if i < 20:
            donchian_high[i] = np.max(high[max(0, i-19):i+1])
            donchian_low[i] = np.min(low[max(0, i-19):i+1])
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + price above EMA50 (uptrend) + volume spike
            if (close[i] > upper_channel and 
                close[i] > ema_trend and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + price below EMA50 (downtrend) + volume spike
            elif (close[i] < lower_channel and 
                  close[i] < ema_trend and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian channel
            if close[i] < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian channel
            if close[i] > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_EMAFilter"
timeframe = "12h"
leverage = 1.0