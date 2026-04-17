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
    
    # === 12h Donchian(20) for structure ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper/lower bands
    upper_20 = np.full_like(high_12h, np.nan)
    lower_20 = np.full_like(low_12h, np.nan)
    for i in range(len(high_12h)):
        if i >= 19:
            upper_20[i] = np.max(high_12h[i-19:i+1])
            lower_20[i] = np.min(low_12h[i-19:i+1])
        else:
            upper_20[i] = np.max(high_12h[0:i+1]) if i > 0 else high_12h[0]
            lower_20[i] = np.min(low_12h[0:i+1]) if i > 0 else low_12h[0]
    
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # === 1d ATR(14) for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 4h Volume spike confirmation ===
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1]) if i > 0 else volume[0]
    vol_spike = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    warmup = 50
    position = 0
    
    for i in range(warmup, n):
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(atr_14_aligned[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volatility filter: ATR > 0.5% of price
        vol_filter = atr_14_aligned[i] > 0.005 * close[i]
        
        if position == 0:
            # Long: break above upper band with volume spike
            if close[i] > upper_aligned[i] and vol_spike[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike
            elif close[i] < lower_aligned[i] and vol_spike[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below lower band
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above upper band
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "Donchian12_VolumeSpike_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0