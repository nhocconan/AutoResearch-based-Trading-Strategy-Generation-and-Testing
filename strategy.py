#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
- Uses 6h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Williams Alligator: Jaw (13-period SMMA, offset 8), Teeth (8-period SMMA, offset 5), Lips (5-period SMMA, offset 3)
- Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5 * 20-period volume MA
- Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5 * 20-period volume MA
- Exit: reverse signal or when Alligator lines cross in opposite direction
- Discrete signal size: 0.25 to balance return and risk
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Alligator catches trends in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    # First value is simple SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
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
    
    # Williams Alligator on 6h timeframe
    # Jaw: 13-period SMMA of median price, offset 8 bars
    # Teeth: 8-period SMMA of median price, offset 5 bars
    # Lips: 5-period SMMA of median price, offset 3 bars
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply offsets (shift right by offset bars)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 50)  # Need Alligator jaws (13+8 offset), volume MA(20), 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND uptrend AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND downtrend AND volume confirmation
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal (bearish alignment) or loss of uptrend
            if lips[i] < teeth[i] or teeth[i] < jaw[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal (bullish alignment) or loss of downtrend
            if lips[i] > teeth[i] or teeth[i] > jaw[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0