#!/usr/bin/env python3
# 4h_VolumeSpike_Reversal_1dTrend
# Hypothesis: Mean reversion after volume spikes in strong daily trends. In bull markets, buy dips with volume spikes; in bear markets, sell rallies with volume spikes. Uses 1d EMA50 for trend and volume/volume_ma > 2.0 for spike detection. Low-frequency signals to avoid fee drag.

name = "4h_VolumeSpike_Reversal_1dTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # In uptrend: buy the dip on volume spike
            if close[i] > ema_50_1d_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # In downtrend: sell the rally on volume spike
            elif close[i] < ema_50_1d_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend breaks or volume spike fades
            if close[i] < ema_50_1d_aligned[i] or volume_ratio[i] < 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend breaks or volume spike fades
            if close[i] > ema_50_1d_aligned[i] or volume_ratio[i] < 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals