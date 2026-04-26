#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_TrendFilter
Hypothesis: On 6h timeframe, combining ADX(14) for trend strength with Williams Alligator (SMAs of median price) as a dynamic trend filter reduces whipsaws in ranging markets while capturing strong trends in both bull and bear markets. Uses 1d HTF for Alligator alignment to avoid lower timeframe noise. Targets 12-25 trades/year via strict ADX > 25 threshold and Alligator alignment requirement. Discrete position sizing (0.0, ±0.25) minimizes fee churn.
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
    volume = prices['volume'].values
    
    # Get 1d data for HTF Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1d: SMAs of median price
    median_price_1d = (high_1d + low_1d) / 2
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h timeframe (no extra delay needed for SMAs)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate ADX(14) on 6h for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero when both DI are zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for all indicators
    start_idx = max(60, 30, 20)  # 1d lookback, Alligator, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx[i])):
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
        adx_val = adx[i]
        close_val = close[i]
        
        # Trend filter: Alligator alignment (Lips > Teeth > Jaw = uptrend, reverse = downtrend)
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        # ADX threshold for strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: bullish Alligator alignment + strong trend
            if bullish_alignment and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: bearish Alligator alignment + strong trend
            elif bearish_alignment and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Loss of bullish Alligator alignment
            if not bullish_alignment:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Weakening trend (ADX < 20)
            elif adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Loss of bearish Alligator alignment
            if not bearish_alignment:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Weakening trend (ADX < 20)
            elif adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_ADX_WilliamsAlligator_TrendFilter"
timeframe = "6h"
leverage = 1.0