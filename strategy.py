#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1w EMA50 trend filter + volume confirmation.
- Williams Alligator: Jaw (EMA13, 8-shift), Teeth (EMA8, 5-shift), Lips (EMA5, 3-shift)
- Long: Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA50 AND volume > 1.5x 20-period avg
- Short: Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA50 AND volume > 1.5x 20-period avg
- Exit: Opposite Alligator alignment OR close crosses 1w EMA50
- Uses 1w HTF for EMA50 trend filter (more stable than 1d for 6h timeframe)
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Alligator catches trends early; weekly EMA filters counter-trend noise
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator from 6h data
    # Jaw: EMA13, 8-shift
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift right by 8 bars (future becomes past)
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: EMA8, 5-shift
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift right by 5 bars
    teeth[:5] = np.nan
    
    # Lips: EMA5, 3-shift
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift right by 3 bars
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need 50 for weekly EMA, 20 for volume MA, 13 for Alligator jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish Alligator AND price > 1w EMA50 AND volume confirmation
            if bullish_alignment and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND price < 1w EMA50 AND volume confirmation
            elif bearish_alignment and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator OR price < 1w EMA50 (trend flip)
            if bearish_alignment or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator OR price > 1w EMA50 (trend flip)
            if bullish_alignment or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_Trend_1wEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0