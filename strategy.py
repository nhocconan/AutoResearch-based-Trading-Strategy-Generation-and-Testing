#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift)
- Long: Lips > Teeth > Jaw (bullish alignment) + price > 1w EMA50 (uptrend) + volume > 1.5x 20-period avg
- Short: Lips < Teeth < Jaw (bearish alignment) + price < 1w EMA50 (downtrend) + volume > 1.5x 20-period avg
- Exit: Alligator lines cross (Lips crosses Teeth) or price crosses 1w EMA50
- Uses 1d timeframe with 1w HTF for trend filter to reduce noise and avoid SOL-only bias
- Volume confirmation ensures breakout validity
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return source.copy()
    smma = np.full_like(source, np.nan, dtype=float)
    smma[length-1] = np.nanmean(source[:length])
    for i in range(length, len(source)):
        smma[i] = (smma[i-1] * (length-1) + source[i]) / length
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set NaN for shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13+8, 50)  # Volume MA, Jaw shift, 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(jaw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Bullish Alligator: Lips > Teeth > Jaw
            bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator: Lips < Teeth < Jaw  
            bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Bullish Alligator + price > 1w EMA50 (uptrend) + volume spike
            if volume_spike and bullish_alligator and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + price < 1w EMA50 (downtrend) + volume spike
            elif volume_spike and bearish_alligator and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator lines cross (Lips crosses below Teeth) or price < 1w EMA50
            if lips[i] < teeth[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator lines cross (Lips crosses above Teeth) or price > 1w EMA50
            if lips[i] > teeth[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0