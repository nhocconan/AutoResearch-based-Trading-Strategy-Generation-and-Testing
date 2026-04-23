#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume spike confirmation.
- Williams Alligator: Jaw (13-period SMA, 8-bar offset), Teeth (8-period SMA, 5-bar offset), Lips (5-period SMA, 3-bar offset)
- Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x 20-period avg
- Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x 20-period avg
- Exit: Opposite Alligator alignment OR price crosses 1d EMA50
- Uses 1d HTF for EMA50 (trend) and volume confirmation uses 6h data
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Alligator catches trends early; EMA50 filters counter-trend whipsaws in bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (all calculated on 6h data)
    # Jaw: 13-period SMA, 8 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, 5 bars ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, 3 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # EMA50, vol MA, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams Alligator alignment
        bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish Alligator AND price > 1d EMA50 AND volume confirmation
            if bullish_align and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND price < 1d EMA50 AND volume confirmation
            elif bearish_align and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator OR price < 1d EMA50 (trend flip)
            if bearish_align or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator OR price > 1d EMA50 (trend flip)
            if bullish_align or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0