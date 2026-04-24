#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Uses 1d timeframe (primary) and 1w HTF for EMA50 trend alignment
- Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period)
- Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment)
- Trend filter: only long when price > 1w EMA50, only short when price < 1w EMA50
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid low-volume false signals
- Discrete signal size: 0.25 to balance return and risk
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Alligator catches trends in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 1d data
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: 13-period smoothed median, offset 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # offset 8 bars into future
    jaw_values = jaw.values
    
    # Teeth: 8-period smoothed median, offset 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # offset 5 bars into future
    teeth_values = teeth.values
    
    # Lips: 5-period smoothed median, offset 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # offset 3 bars into future
    lips_values = lips.values
    
    # Align Alligator components to LTF
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw_values)  # same timeframe, just offset
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth_values)
    lips_aligned = align_htf_to_ltf(prices, prices, lips_values)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # Need EMA50, volume MA, and Alligator offsets
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND uptrend AND volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND downtrend AND volume confirmation
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment or reverse signal
            if lips_aligned[i] < teeth_aligned[i] or lips_aligned[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment or reverse signal
            if lips_aligned[i] > teeth_aligned[i] or lips_aligned[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0