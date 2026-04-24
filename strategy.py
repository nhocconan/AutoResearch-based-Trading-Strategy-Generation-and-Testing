#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
- 1d EMA50 ensures alignment with daily trend to reduce whipsaws in ranging markets.
- Volume spike (>1.8x 30-bar average) confirms participation.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 60-120 total over 4 years (15-30/year) to minimize fee drag.
- Works in bull/bear markets via daily trend filter and Alligator's trend identification.
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
    
    # Get 1d data ONCE before loop for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (primary)
    # Jaw: 13-period SMMA, offset 8 bars
    # Teeth: 8-period SMMA, offset 5 bars  
    # Lips: 5-period SMMA, offset 3 bars
    def smma(src, length):
        """Smoothed Moving Average"""
        result = np.full_like(src, np.nan, dtype=np.float64)
        if len(src) < length:
            return result
        # First value is SMA
        result[length-1] = np.mean(src[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CLOSE) / LENGTH
        for i in range(length, len(src)):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set offset bars to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 13+8) + 1  # Need enough for EMA, volume, Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Alligator alignment for uptrend: Lips > Teeth > Jaw
                if lips[i] > teeth[i] > jaw[i]:
                    # Additional filter: price above 1d EMA50 for long bias
                    if close[i] > ema_50_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                # Alligator alignment for downtrend: Jaw > Teeth > Lips
                elif jaw[i] > teeth[i] > lips[i]:
                    # Additional filter: price below 1d EMA50 for short bias
                    if close[i] < ema_50_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Alligator loses alignment (Lips < Teeth OR Teeth < Jaw) OR price crosses below 1d EMA50
            if lips[i] < teeth[i] or teeth[i] < jaw[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses alignment (Jaw < Teeth OR Teeth < Lips) OR price crosses above 1d EMA50
            if jaw[i] < teeth[i] or teeth[i] < lips[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0