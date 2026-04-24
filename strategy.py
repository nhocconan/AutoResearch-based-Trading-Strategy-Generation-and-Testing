#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Uses Williams Alligator (Jaw, Teeth, Lips) from 1d timeframe to identify trend.
- Jaw (13-period SMMA smoothed 8 bars), Teeth (8-period SMMA smoothed 5 bars), Lips (5-period SMMA smoothed 3 bars).
- Long when Lips > Teeth > Jaw (bullish alignment) and price > Jaw.
- Short when Lips < Teeth < Jaw (bearish alignment) and price < Jaw.
- Trend filter: price must be above/below 1w EMA50 to align with higher timeframe direction.
- Volume confirmation: volume > 1.5x 20-bar average to avoid false breakouts.
- Designed for 1d timeframe to capture medium-term trends with low trade frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 7-25 trades/year (30-100 total over 4 years) to stay fee-efficient.
- Williams Alligator is effective in both trending and ranging markets due to its multiple smoothed averages.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    alpha = 1.0 / length
    # First value is simple average
    if not np.isnan(source[0]):
        result[0] = source[0]
    else:
        result[0] = np.nan
    for i in range(1, len(source)):
        if np.isnan(source[i]) or np.isnan(result[i-1]):
            result[i] = np.nan
        else:
            result[i] = result[i-1] + alpha * (source[i] - result[i-1])
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Alligator calculations
        return np.zeros(n)
    
    # Calculate Williams Alligator components for 1d timeframe
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Jaw: 13-period SMMA, smoothed 8 bars
    jaw_raw = smma(median_price_1d, 13)
    jaw = smma(jaw_raw, 8)
    
    # Teeth: 8-period SMMA, smoothed 5 bars
    teeth_raw = smma(median_price_1d, 8)
    teeth = smma(teeth_raw, 5)
    
    # Lips: 5-period SMMA, smoothed 3 bars
    lips_raw = smma(median_price_1d, 5)
    lips = smma(lips_raw, 3)
    
    # Align Alligator lines to 1d timeframe (wait for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Williams Alligator signals
        lips_gt_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_gt_jaw = teeth_aligned[i] > jaw_aligned[i]
        bullish_alignment = lips_gt_teeth and teeth_gt_jaw
        
        lips_lt_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_lt_jaw = teeth_aligned[i] < jaw_aligned[i]
        bearish_alignment = lips_lt_teeth and teeth_lt_jaw
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Long: bullish alignment AND price above Jaw AND above 1w EMA50
                if bullish_alignment and close[i] > jaw_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish alignment AND price below Jaw AND below 1w EMA50
                elif bearish_alignment and close[i] < jaw_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bearish alignment OR price below Jaw OR below 1w EMA50
            if (not bullish_alignment) or close[i] < jaw_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR price above Jaw OR above 1w EMA50
            if (not bearish_alignment) or close[i] > jaw_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0