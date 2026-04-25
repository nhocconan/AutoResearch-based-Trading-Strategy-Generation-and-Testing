#!/usr/bin/env python3
"""
6h_Williams_Alligator_Trend_1dRegimeFilter_v1
Hypothesis: Williams Alligator (jaw/teeth/lips) on 6h defines trend, filtered by 1d ADX regime (ADX>25 = trend, ADX<20 = range). Only trade in Alligator alignment direction during trending regimes to avoid whipsaw. Uses discrete sizing (0.25) and time-based exit (8 bars) to limit overtrading. Targets 12-37 trades/year on 6h timeframe.
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
    
    # Get 6h data for Williams Alligator - primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars  
    # Lips: 5-period SMA, shifted 3 bars
    median_price = (high_6h + low_6h) / 2
    
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan
    
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan
    
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data for ADX regime filter - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need for ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 30)  # Alligator needs ~50, ADX needs 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        
        # Get 6h close aligned for direct comparison
        close_6h_aligned = align_htf_to_ltf(prices, df_6h, close_6h)
        close_6h_val = close_6h_aligned[i]
        
        # Alligator trend alignment: 
        # Bullish: lips > teeth > jaw (all aligned upward)
        # Bearish: lips < teeth < jaw (all aligned downward)
        bullish_align = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_align = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Look for entry signals: Alligator alignment in trending regime only
            long_signal = bullish_align and is_trending
            short_signal = bearish_align and is_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit conditions:
            # 1. Time-based exit: 8 bars to avoid overtrading
            # 2. Alligator loses alignment (lips crosses below teeth)
            if bars_since_entry >= 8:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif not (lips_val > teeth_val):  # lips crossed below teeth
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit conditions:
            # 1. Time-based exit: 8 bars to avoid overtrading
            # 2. Alligator loses alignment (lips crosses above teeth)
            if bars_since_entry >= 8:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif not (lips_val < teeth_val):  # lips crossed above teeth
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "6h_Williams_Alligator_Trend_1dRegimeFilter_v1"
timeframe = "6h"
leverage = 1.0