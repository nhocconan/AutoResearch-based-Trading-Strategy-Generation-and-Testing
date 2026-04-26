#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_Trend
Hypothesis: Combine ADX(14) for trend strength with Williams Alligator (SMAs 13,8,5) on 6h to filter trend direction.
Enter long when ADX>25 (strong trend) and Alligator aligned bullish (jaw>teeth>lips).
Enter short when ADX>25 and Alligator aligned bearish (jaw<teeth<lips).
Exit when ADX<20 (trend weakens) or Alligator alignment breaks.
Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Target: 50-120 trades over 4 years (12-30/year) to minimize fee drag.
Works in bull (captures strong uptrends) and bear (captures strong downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for indicators (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) on 6h
    # True Range
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr = np.concatenate([[0], tr1])
    # Plus Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    # Minus Directional Movement
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM with Welles Wilder smoothing (alpha=1/period)
    def wilde_smooth(arr, period):
        res = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return res
        res[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            res[i] = res[i-1] - (res[i-1] / period) + arr[i]
        return res
    
    atr_6h = wilde_smooth(tr, 14)
    plus_di_6h = 100 * wilde_smooth(plus_dm, 14) / atr_6h
    minus_di_6h = 100 * wilde_smooth(minus_dm, 14) / atr_6h
    dx_6h = 100 * np.abs(plus_di_6h - minus_di_6h) / (plus_di_6h + minus_di_6h)
    adx_6h = wilde_smooth(dx_6h, 14)
    
    # Calculate Williams Alligator on 6h (SMAs 13,8,5)
    jaw_6h = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_6h = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_6h = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 6h indicators to LTF
    adx_6h_aligned = align_htf_to_ltf(prices, df_6h, adx_6h)
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of ADX(14) smoothing, Alligator jaws(13), EMA50(1d)
    start_idx = max(14+13, 13, 50) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_6h_aligned[i]) or
            np.isnan(jaw_6h_aligned[i]) or
            np.isnan(teeth_6h_aligned[i]) or
            np.isnan(lips_6h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        adx_val = adx_6h_aligned[i]
        jaw_val = jaw_6h_aligned[i]
        teeth_val = teeth_6h_aligned[i]
        lips_val = lips_6h_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        
        # Alligator alignment
        bullish_aligned = (jaw_val > teeth_val) and (teeth_val > lips_val)
        bearish_aligned = (jaw_val < teeth_val) and (teeth_val < lips_val)
        
        # Trend filter: price above/below 1d EMA50
        uptrend_filter = close_val > ema_50_val
        downtrend_filter = close_val < ema_50_val
        
        if position == 0:
            # Long: strong ADX + bullish Alligator + uptrend filter
            long_signal = (adx_val > 25) and bullish_aligned and uptrend_filter
            
            # Short: strong ADX + bearish Alligator + downtrend filter
            short_signal = (adx_val > 25) and bearish_aligned and downtrend_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ADX weakens OR Alligator alignment breaks OR trend filter fails
            if (adx_val < 20) or not bullish_aligned or not uptrend_filter:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ADX weakens OR Alligator alignment breaks OR trend filter fails
            if (adx_val < 20) or not bearish_aligned or not downtrend_filter:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_WilliamsAlligator_Trend"
timeframe = "6h"
leverage = 1.0