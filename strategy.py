#!/usr/bin/env python3
"""
4h_WilliamsAlligator_Trend_WithVolume
Strategy: Williams Alligator trend filter on 4h with 1d confirmation and volume spike.
Long: 4h price > Alligator Lips (13) > Teeth (8) > Jaw (13) AND 1d close > 1d EMA50 AND volume > 1.5x 4h volume MA20
Short: 4h price < Alligator Lips < Teeth < Jaw AND 1d close < 1d EMA50 AND volume > 1.5x 4h volume MA20
Exit: Price crosses back below Lips (long) or above Lips (short)
Position size: 0.25
Williams Alligator uses SMMA (smoothed moving average) to avoid whipsaw in sideways markets.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 4h (Jaw=13, Teeth=8, Lips=5) - using SMMA
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA50 on 1d
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 1d EMA50 to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 4h volume > 1.5x 20-period MA
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need enough for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # 1d trend filter
        price_above_1d_ema = close[i] > ema50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema50_1d_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_alignment = (lips[i] > teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i] < jaw[i])
        
        if position == 0:
            # Long: bullish alignment + volume + 1d uptrend
            if bullish_alignment and volume_filter and price_above_1d_ema:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + volume + 1d downtrend
            elif bearish_alignment and volume_filter and price_below_1d_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below Lips (Alligator wake up signal)
            if close[i] < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above Lips
            if close[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_Trend_WithVolume"
timeframe = "4h"
leverage = 1.0