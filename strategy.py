#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_ADX_v1
Breakout of 20-bar Donchian channel with volume spike and ADX trend filter.
Long when price breaks above upper band with volume > 1.5x average and ADX > 25.
Short when price breaks below lower band with volume > 1.5x average and ADX > 25.
Exit when price returns to the middle of the channel or ADX < 20.
Designed to capture strong momentum moves with volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_channel = (highest_high + lowest_low) / 2.0
    
    # === Volume Spike (1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === ADX(14) for trend strength ===
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper channel + volume spike + ADX > 25
            if (close[i] > highest_high[i] and 
                volume_spike[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower channel + volume spike + ADX > 25
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to middle channel OR ADX < 20
            if (close[i] <= middle_channel[i] or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle channel OR ADX < 20
            if (close[i] >= middle_channel[i] or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADX_v1"
timeframe = "4h"
leverage = 1.0