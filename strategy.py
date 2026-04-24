#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
- Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trend absence/presence.
- Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x 20-bar average.
- Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x 20-bar average.
- Uses 6h timeframe to balance signal quality and trade frequency (~12-37 trades/year).
- Position size 0.25 limits drawdown. Volume confirmation reduces false signals in chop.
- Works in bull/bear via trend filter and Alligator's trend-defining capability.
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
    
    # Williams Alligator on 6h: Jaw(13), Teeth(8), Lips(5) - all SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)  # Need enough for EMA, Jaw, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    # Long: price above 1d EMA50
                    if close[i] > ema_50_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    # Short: price below 1d EMA50
                    if close[i] < ema_50_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks bearish OR price crosses below EMA
            if (lips[i] < teeth[i] or teeth[i] < jaw[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks bullish OR price crosses above EMA
            if (lips[i] > teeth[i] or teeth[i] > jaw[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0