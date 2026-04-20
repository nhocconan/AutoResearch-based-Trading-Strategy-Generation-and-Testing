#!/usr/bin/env python3
"""
6h_12h_ADX_Trend_1d_Consolidation
Concept: Trend following on 6h with 12h ADX trend filter and 1d consolidation filter.
- Long: Price > 6h EMA(20) AND 12h ADX(14) > 25 AND 1d Bollinger Band Width < 0.5 (consolidation)
- Short: Price < 6h EMA(20) AND 12h ADX(14) > 25 AND 1d Bollinger Band Width < 0.5
- Exit: Price crosses 6h EMA(20) in opposite direction
- Position sizing: 0.25
- Works in bull/bear: ADX filters strong trends, consolidation filter avoids chop
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_ADX_Trend_1d_Consolidation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 6h: EMA(20) for trend following ===
    close = prices['close'].values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 12h: ADX(14) trend strength filter ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([ [np.nan], tr ])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([ [0], dm_plus ])
    dm_minus = np.concatenate([ [0], dm_minus ])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])  # skip index 0
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align 12h ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 1d: Bollinger Band Width consolidation filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical Price
    tp = (high_1d + low_1d + close_1d) / 3.0
    # TP SMA(20)
    tp_ma = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    # TP Standard Deviation(20)
    tp_std = pd.Series(tp).rolling(window=20, min_periods=20).std().values
    # Bollinger Bands
    bb_upper = tp_ma + 2 * tp_std
    bb_lower = tp_ma - 2 * tp_std
    # Band Width
    bb_width = (bb_upper - bb_lower) / tp_ma
    # Align to 6h
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema = ema_20[i]
        adx = adx_aligned[i]
        bbw = bb_width_aligned[i]
        price = close[i]
        
        # Skip if any value is NaN
        if np.isnan(ema) or np.isnan(adx) or np.isnan(bbw):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend and consolidation conditions
        trend_strong = adx > 25
        consolidating = bbw < 0.5
        
        if position == 0:
            # Long: price above EMA, strong trend, consolidation
            if price > ema and trend_strong and consolidating:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA, strong trend, consolidation
            elif price < ema and trend_strong and consolidating:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA
            if price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA
            if price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals