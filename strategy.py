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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for Williams Alligator) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Williams Alligator (1d) ===
    # Jaw: 13-period smoothed, 8 periods future
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period smoothed, 5 periods future
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period smoothed, 3 periods future
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 6t
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 6x EMA (trend filter) ===
    ema_6_6h = pd.Series(close_6h).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # === 6x Volume ratio for confirmation ===
    vol_ma_6_6h = pd.Series(volume_6h).rolling(window=6, min_periods=6).mean().values
    vol_ratio_6h = volume_6h / vol_ma_6_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_6_6h[i]) or 
            np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_6_6h[i]
        vol_ratio = vol_ratio_6h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: Alligator lines cross (lips below teeth) OR trend weakens
            if lips_val < teeth_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (lips above teeth) OR trend weakens
            if lips_val > teeth_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) with volume, above EMA
            if lips_val > teeth_val and teeth_val > jaw_val and vol_ratio > 1.3 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Lips < Teeth < Jaw (bearish alignment) with volume, below EMA
            elif lips_val < teeth_val and teeth_val < jaw_val and vol_ratio > 1.3 and price < ema_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_EMA_Volume"
timeframe = "6h"
leverage = 1.0