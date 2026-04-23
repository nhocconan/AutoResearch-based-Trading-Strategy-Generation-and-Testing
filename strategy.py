#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1w EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period smoothed median), Lips (5-period smoothed median)
- Long: Lips > Teeth > Jaw (bullish alignment) + Close > Lips + volume > 1.5x 20-period avg + price > 1w EMA50
- Short: Lips < Teeth < Jaw (bearish alignment) + Close < Lips + volume > 1.5x 20-period avg + price < 1w EMA50
- Exit: Opposite Alligator alignment or EMA50 trend flip
- Uses Alligator for trend structure, volume for conviction, 1w EMA50 for HTF trend filter
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (trend alignment with price above/below Alligator) and bear markets (trend filtered by 1w EMA50)
"""

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
    
    # Volume confirmation: > 1.5x 20-period average (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator components (using median price)
    median_price = (high + low) / 2
    
    # Jaw: 13-period smoothed median (8 periods offset)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().shift(8)
    jaw_values = jaw.values
    
    # Teeth: 8-period smoothed median (5 periods offset)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().shift(5)
    teeth_values = teeth.values
    
    # Lips: 5-period smoothed median (3 periods offset)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().shift(3)
    lips_values = lips.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # Need 50 for EMA, 20 for volume, Alligator offsets
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(jaw_values[i]) or
            np.isnan(teeth_values[i]) or
            np.isnan(lips_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator alignment
        bullish_alignment = lips_values[i] > teeth_values[i] and teeth_values[i] > jaw_values[i]
        bearish_alignment = lips_values[i] < teeth_values[i] and teeth_values[i] < jaw_values[i]
        
        if position == 0:
            # Long: Bullish alignment + Close > Lips + volume confirmation + price > 1w EMA50
            if (bullish_alignment and 
                close[i] > lips_values[i] and 
                volume_confirm and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + Close < Lips + volume confirmation + price < 1w EMA50
            elif (bearish_alignment and 
                  close[i] < lips_values[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish alignment OR price < 1w EMA50 (trend flip)
            if bearish_alignment or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alignment OR price > 1w EMA50 (trend flip)
            if bullish_alignment or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0