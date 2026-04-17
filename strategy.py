#!/usr/bin/env python3
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
    
    # === 1d Donchian Channels (20-period) for price channel structure ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian upper and lower bands
    upper_20 = np.full_like(high_1d, np.nan)
    lower_20 = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            upper_20[i] = np.max(high_1d[i-19:i+1])
            lower_20[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            upper_20[i] = np.max(high_1d[0:i+1])
            lower_20[i] = np.min(low_1d[0:i+1])
        else:
            upper_20[i] = high_1d[0]
            lower_20[i] = low_1d[0]
    
    # === 1d ATR (14-period) for volatility filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === 1d Volume average (20-period) for volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_1d[0:i+1])
        else:
            vol_ma_20[i] = volume_1d[0]
    
    # Align all indicators to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = volume_1d[i] > vol_ma_20_aligned[i] * 1.5
        
        # Entry logic: only enter when flat AND in high volume regime
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation
            if (close[i] > upper_20_aligned[i] and 
                atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                vol_confirm):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian lower + volume confirmation
            elif (close[i] < lower_20_aligned[i] and 
                  atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below Donchian lower
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian upper
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeConfirm_VolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0