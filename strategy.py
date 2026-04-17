#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Williams Alligator (13,8,5 SMAs) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Jaw (13-period SMA, shifted 8 bars)
    jaw = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 12:
            jaw[i] = np.mean(close_1d[i-12:i+1])
    
    # Teeth (8-period SMA, shifted 5 bars)
    teeth = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 7:
            teeth[i] = np.mean(close_1d[i-7:i+1])
    
    # Lips (5-period SMA, shifted 3 bars)
    lips = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 4:
            lips[i] = np.mean(close_1d[i-4:i+1])
    
    # Shift jaws/teeth/lips for Alligator alignment
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) >= 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) >= 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) >= 3:
        lips_shifted[3:] = lips[:-3]
    
    # === 1d ATR (14-period) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
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
    
    # === 12h Volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume on 12h timeframe
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_12h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_12h[0]
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average
    vol_confirm = volume_12h > vol_ma_20 * 1.5
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator signals: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: Bullish Alligator alignment + volatility filter + volume confirmation
            if (bullish and 
                atr_14_aligned[i] > 0.003 * close[i] and  # volatility filter
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Bearish Alligator alignment + volatility filter + volume confirmation
            elif (bearish and 
                  atr_14_aligned[i] > 0.003 * close[i] and  # volatility filter
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse Alligator signal or volatility collapse
        elif position == 1:
            # Exit long: bearish alignment OR volatility too low
            if bearish or atr_14_aligned[i] <= 0.003 * close[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment OR volatility too low
            if bullish or atr_14_aligned[i] <= 0.003 * close[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Volume_VolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0