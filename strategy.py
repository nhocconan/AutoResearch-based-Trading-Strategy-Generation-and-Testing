#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_Trend
Hypothesis: Combines ADX trend strength with Williams Alligator (three SMAs) to identify strong trends.
Uses 1d timeframe for ADX and Alligator to filter 6s entries. Works in bull/bear by only taking trades
when ADX > 25 (strong trend) and price is aligned with Alligator direction (bullish/bearish alignment).
Targets 15-30 trades/year on 6s timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _sma(arr, period):
    """Simple moving average with NaN for insufficient data."""
    if len(arr) < period:
        return np.full(len(arr), np.nan)
    sma = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        sma[i] = np.mean(arr[i - period + 1:i + 1])
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ADX and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (period=14)
    def _smooth(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        smoothed = np.full(len(arr), np.nan)
        smoothed[period-1] = np.nansum(arr[1:period])  # skip index 0 for DM
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr14 = _smooth(tr, 14)
    plus_dm14 = _smooth(plus_dm, 14)
    minus_dm14 = _smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = _smooth(dx, 14)  # ADX is smoothed DX
    
    # Williams Alligator (three SMAs: 13, 8, 5 with shifts 8, 5, 3)
    # Jaw (blue): 13-period SMA, shifted 8 bars
    ma13 = _sma(close_1d, 13)
    jaw = np.roll(ma13, 8)  # shift forward 8 bars
    jaw[:8] = np.nan
    
    # Teeth (red): 8-period SMA, shifted 5 bars
    ma8 = _sma(close_1d, 8)
    teeth = np.roll(ma8, 5)  # shift forward 5 bars
    teeth[:5] = np.nan
    
    # Lips (green): 5-period SMA, shifted 3 bars
    ma5 = _sma(close_1d, 5)
    lips = np.roll(ma5, 3)  # shift forward 3 bars
    lips[:3] = np.nan
    
    # Align 1d indicators to 6s timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Lips < Teeth < Jaw (green < red < blue)
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + bullish Alligator alignment
            if adx_aligned[i] > 25 and bullish_aligned:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend) + bearish Alligator alignment
            elif adx_aligned[i] > 25 and bearish_aligned:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: ADX < 20 (weakening trend) or Alligator alignment breaks
            if adx_aligned[i] < 20 or not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: ADX < 20 (weakening trend) or Alligator alignment breaks
            if adx_aligned[i] < 20 or not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_Trend"
timeframe = "6h"
leverage = 1.0