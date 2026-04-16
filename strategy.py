#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw=13, Teeth=8, Lips=5) with 1d EMA50 trend filter and volume confirmation
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-period avg volume
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-period avg volume
# Williams Alligator identifies trending vs ranging markets - effective in both bull and bear regimes
# EMA50 filter ensures alignment with daily trend, reducing counter-trend trades
# Volume confirmation adds conviction to signals
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Williams Alligator (13,8,5) on 12h timeframe ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Jaw (13-period SMMA)
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMMA)
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMMA)
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for confirmation
        
        # === EXIT LOGIC (reverse signal) ===
        if position == 1:  # Long position
            # Exit when bearish alignment OR price < EMA50
            if lips_val < teeth_val or teeth_val < jaw_val or price < ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when bullish alignment OR price > EMA50
            if lips_val > teeth_val or teeth_val > jaw_val or price > ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = lips_val > teeth_val and teeth_val > jaw_val
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = lips_val < teeth_val and teeth_val < jaw_val
            
            # Long when: bullish alignment AND price > EMA50 AND volume confirmation
            if bullish and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: bearish alignment AND price < EMA50 AND volume confirmation
            elif bearish and price < ema_val and vol_confirm:
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

name = "12h_WilliamsAlligator_1dEMA50_Volume1.5x"
timeframe = "12h"
leverage = 1.0