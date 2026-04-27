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
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (Jaws, Teeth, Lips) - 13,8,5 SMAs with shifts
    # Jaws: 13-period SMA, shifted 8 bars forward
    jaw_period = 13
    jaw_shift = 8
    sma_jaw = np.full(len(close_1d), np.nan)
    if len(close_1d) >= jaw_period:
        for i in range(jaw_period - 1, len(close_1d)):
            sma_jaw[i] = np.mean(close_1d[i - jaw_period + 1:i + 1])
    jaw = np.full(len(close_1d), np.nan)
    if len(close_1d) >= jaw_period + jaw_shift:
        for i in range(jaw_shift, len(close_1d)):
            jaw[i] = sma_jaw[i - jaw_shift]
    
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth_period = 8
    teeth_shift = 5
    sma_teeth = np.full(len(close_1d), np.nan)
    if len(close_1d) >= teeth_period:
        for i in range(teeth_period - 1, len(close_1d)):
            sma_teeth[i] = np.mean(close_1d[i - teeth_period + 1:i + 1])
    teeth = np.full(len(close_1d), np.nan)
    if len(close_1d) >= teeth_period + teeth_shift:
        for i in range(teeth_shift, len(close_1d)):
            teeth[i] = sma_teeth[i - teeth_shift]
    
    # Lips: 5-period SMA, shifted 3 bars forward
    lips_period = 5
    lips_shift = 3
    sma_lips = np.full(len(close_1d), np.nan)
    if len(close_1d) >= lips_period:
        for i in range(lips_period - 1, len(close_1d)):
            sma_lips[i] = np.mean(close_1d[i - lips_period + 1:i + 1])
    lips = np.full(len(close_1d), np.nan)
    if len(close_1d) >= lips_period + lips_shift:
        for i in range(lips_shift, len(close_1d)):
            lips[i] = sma_lips[i - lips_shift]
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    # Calculate 20-period average volume on 12h
    avg_vol_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        for i in range(19, len(volume_12h)):
            avg_vol_12h[i] = np.mean(volume_12h[i - 19:i + 1])
    
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator lines and volume average
    start_idx = max(34, 20)  # Need enough data for Alligator and volume average
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(avg_vol_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_vol_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average 12h volume
        vol_confirm = vol > 1.5 * avg_vol
        
        # Williams Alligator signals:
        # Bullish alignment: Lips > Teeth > Jaws (alligator eating with mouth up)
        # Bearish alignment: Lips < Teeth < Jaws (alligator eating with mouth down)
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: Bullish alignment + volume confirmation
            if bullish_alignment and vol_confirm:
                signals[i] = size
                position = 1
            # Short: Bearish alignment + volume confirmation
            elif bearish_alignment and vol_confirm:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: When alignment breaks (lips crosses below teeth)
            if lips_val <= teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: When alignment breaks (lips crosses above teeth)
            if lips_val >= teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6H_Williams_Alligator_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0