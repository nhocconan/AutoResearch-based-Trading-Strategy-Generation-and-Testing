#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Long when Lips > Teeth > Jaw (bullish alignment) with volume confirmation
- Short when Lips < Teeth < Jaw (bearish alignment) with volume confirmation
- Trend filter: only long when price > 1d EMA50, only short when price < 1d EMA50
- Volume confirmation: current volume > 1.5 * 20-period volume MA
- Discrete signal size: 0.25 to balance return and risk
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Alligator catches trends in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also known as RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    alpha = 1.0 / length
    # Initialize with SMA for first value
    if len(source) >= length:
        result[length-1] = np.mean(source[:length])
        for i in range(length, len(source)):
            if not np.isnan(source[i]):
                result[i] = alpha * source[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = result[i-1]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    jaw = smma(close, 13)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    
    teeth = smma(close, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    lips = smma(close, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # Need EMA50, volume MA, and Alligator components
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND uptrend AND volume confirmation
            if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and 
                uptrend[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND downtrend AND volume confirmation
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and 
                  downtrend[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment (Lips < Teeth < Jaw) or reverse signal
            if lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment (Lips > Teeth > Jaw) or reverse signal
            if lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0