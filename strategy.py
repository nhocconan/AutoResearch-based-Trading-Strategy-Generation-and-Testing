#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator trend with 1w EMA50 filter and volume confirmation.
- Long: Jaw < Teeth < Lips (bullish alignment) AND price > 1w EMA50 AND volume > 1.5x 20-period avg
- Short: Jaw > Teeth > Lips (bearish alignment) AND price < 1w EMA50 AND volume > 1.5x 20-period avg
- Exit: Opposite Alligator alignment OR price crosses 1w EMA50
- Uses 1w HTF for EMA50 trend filter to capture major market regime
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Works in bull (buy during bullish alignment above EMA50) and bear (sell during bearish alignment below EMA50)
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
    
    # Volume confirmation: > 1.5x 20-period average (20*12h = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator (SMA with specific periods)
    # Jaw: SMA(13, 8) - 13-period SMA shifted 8 bars ahead
    # Teeth: SMA(8, 5) - 8-period SMA shifted 5 bars ahead  
    # Lips: SMA(5, 3) - 5-period SMA shifted 3 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # Need 50 for EMA, 20 for volume MA, Alligator shifts
    
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
        
        # Alligator alignment signals
        bullish_alignment = jaw[i] < teeth[i] < lips[i]  # Jaw < Teeth < Lips
        bearish_alignment = jaw[i] > teeth[i] > lips[i]  # Jaw > Teeth > Lips
        
        if position == 0:
            # Long: Bullish Alligator alignment AND price > 1w EMA50 AND volume confirmation
            if bullish_alignment and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND price < 1w EMA50 AND volume confirmation
            elif bearish_alignment and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR price < 1w EMA50 (trend flip)
            if bearish_alignment or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR price > 1w EMA50 (trend flip)
            if bullish_alignment or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Trend_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0