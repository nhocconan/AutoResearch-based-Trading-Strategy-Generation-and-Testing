#!/usr/bin/env python3
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
    
    # === 1d Donchian Channels (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate upper and lower bands
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(low_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            upper[i] = np.max(high_1d[i-19:i+1])
            lower[i] = np.min(low_1d[i-19:i+1])
    
    # === 1d ATR (14-period) for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === 12h Volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate 10-period average volume on 12h timeframe
    vol_ma_10 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 9:
            vol_ma_10[i] = np.mean(volume_12h[i-9:i+1])
        else:
            vol_ma_10[i] = volume_12h[i] if i == 0 else np.mean(volume_12h[:i+1])
    
    # Volume confirmation: current 12h volume > 1.3x 10-period average
    vol_confirm = volume_12h > vol_ma_10 * 1.3
    
    # Align all indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: price breaks above upper Donchian band + volatility filter + volume confirmation
            if (close[i] > upper_aligned[i] and 
                atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian band + volatility filter + volume confirmation
            elif (close[i] < lower_aligned[i] and 
                  atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: opposite band touch
        elif position == 1:
            # Exit long: price touches or goes below lower band
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches or goes above upper band
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeConfirm_VolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0