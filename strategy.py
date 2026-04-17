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
    
    # === 1d Williams Alligator (Jaws, Teeth, Lips) ===
    df_1d = get_htf_data(prices, '1d')
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Smoothed medians (SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(median_1d, 13)  # 13-period
    teeth = smma(median_1d, 8)  # 8-period
    lips = smma(median_1d, 5)   # 5-period
    
    # Align to 4h
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 4h Volume Spike (2x 20-period average) ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    volume_spike = volume_4h > (vol_ma_20 * 2.0)
    
    # === 4h ATR Filter (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions: Lips > Teeth > Jaws (bullish alignment) OR Lips < Teeth < Jaws (bearish)
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i]
        
        # Volume and volatility filters
        vol_cond = volume_spike[i]
        vol_filter = atr_14[i] > 0.005 * close[i]  # Minimum volatility
        
        if position == 0:
            # Long entry: bullish alignment + volume spike + volatility
            if bullish_alignment and vol_cond and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short entry: bearish alignment + volume spike + volatility
            elif bearish_alignment and vol_cond and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: alignment breaks
        elif position == 1:
            if not bullish_alignment:  # Bullish alignment broken
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if not bearish_alignment:  # Bearish alignment broken
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Alignment_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0