#!/usr/bin/env python3
"""
6h ADX + Williams Alligator Combination
Hypothesis: Williams Alligator (SMAs with specific periods) identifies trend direction and alignment, while ADX measures trend strength. Long when Alligator is bullish aligned (jaw<teeth<lips) AND ADX > 25, short when bearish aligned (jaw>teeth>lips) AND ADX > 25. Uses 1d Alligator for higher timeframe trend filter and 6h ADX for entry confirmation. Designed for 6h timeframe to target 12-37 trades/year, minimizing fee drag. Works in both bull and bear markets by only taking strong trend signals and avoiding ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator on 1d: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMAs
    # Jaw: 13-period SMMA shifted 8 bars
    jaw_1d = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_1d = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA shifted 3 bars
    lips_1d = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate ADX on 6h (primary timeframe)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(np.roll(close, 1) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations (Alligator: max shift 8, ADX: 14*3 for stability)
    start_idx = max(13+8, 8+5, 5+3, 14*3)  # ~42
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment signals
        bullish_aligned = (jaw_1d_aligned[i] < teeth_1d_aligned[i]) and (teeth_1d_aligned[i] < lips_1d_aligned[i])
        bearish_aligned = (jaw_1d_aligned[i] > teeth_1d_aligned[i]) and (teeth_1d_aligned[i] > lips_1d_aligned[i])
        
        # ADX trend strength filter
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Look for entry signals
            # Long: bullish Alligator alignment AND strong trend
            long_entry = bullish_aligned and strong_trend
            # Short: bearish Alligator alignment AND strong trend
            short_entry = bearish_aligned and strong_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: loss of bullish alignment OR trend weakens (ADX < 20)
            if not bullish_aligned or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: loss of bearish alignment OR trend weakens (ADX < 20)
            if not bearish_aligned or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_1dTrend_AdxFilter"
timeframe = "6h"
leverage = 1.0