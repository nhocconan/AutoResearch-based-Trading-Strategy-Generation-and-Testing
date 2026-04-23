#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (EMA13 smoothed 8), Teeth (EMA8 smoothed 5), Lips (EMA5 smoothed 3)
- Long: Lips > Teeth > Jaw (bullish alignment) + volume > 2.0x 50-period avg + price > 1w EMA50 (uptrend)
- Short: Lips < Teeth < Jaw (bearish alignment) + volume > 2.0x 50-period avg + price < 1w EMA50 (downtrend)
- Exit: Opposite Alligator alignment (Lips <= Teeth for long exit, Lips >= Teeth for short exit)
- 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
- High volume threshold (2.0x) reduces false signals and controls trade frequency
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe
- Works in both bull (trend continuation via Alligator alignment) and bear (avoids counter-trend via weekly filter)
"""

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
    
    # Volume confirmation: > 2.0x 50-period average (strict spike filter)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Williams Alligator components (using 12h data)
    # Jaw: Blue line - 13-period SMMA smoothed by 8 periods
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: Red line - 8-period SMMA smoothed by 5 periods
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips: Green line - 5-period SMMA smoothed by 3 periods
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 8, 5)  # Need 50 for volume MA, 13/8/5 for Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average - strict filter)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish alignment + volume spike + price > 1w EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if bullish_alignment:
                    signals[i] = 0.25
                    position = 1
            # Short: Bearish alignment + volume spike + price < 1w EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if bearish_alignment:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Loss of bullish alignment (Lips <= Teeth)
            if lips[i] <= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Loss of bearish alignment (Lips >= Teeth)
            if lips[i] >= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0