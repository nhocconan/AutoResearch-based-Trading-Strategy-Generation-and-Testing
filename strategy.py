#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses breakout at extreme Camarilla levels (R4/S4) for strong momentum continuation.
- 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws.
- Volume confirmation (>2.0x 24-period average) ensures institutional participation.
- Designed for 12-25 trades/year (50-100 total over 4 years) to minimize fee drag.
- Works in both bull and bear markets by following HTF trend direction.
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (completed daily bar)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 6h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels
    camarilla_h4 = close_1d_aligned + 1.1 * (high_1d_aligned - low_1d_aligned) / 2
    camarilla_l4 = close_1d_aligned - 1.1 * (high_1d_aligned - low_1d_aligned) / 2
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H4 AND price above 1d EMA50 AND volume confirmation
            if close[i] > camarilla_h4[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L4 AND price below 1d EMA50 AND volume confirmation
            elif close[i] < camarilla_l4[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L4 OR price crosses below 1d EMA50
            if close[i] < camarilla_l4[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H4 OR price crosses above 1d EMA50
            if close[i] > camarilla_h4[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4L4_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0