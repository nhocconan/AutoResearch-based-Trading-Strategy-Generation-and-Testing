#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
# Bollinger Band squeeze (low volatility) precedes explosive moves.
# ADX > 25 confirms trend strength for breakout direction.
# Volume spike confirms institutional participation.
# Designed for ~15-25 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    # Middle band (SMA)
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Standard deviation
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    # Upper and lower bands
    upper = basis + dev
    lower = basis - dev
    
    # Bollinger Band Width (normalized)
    bb_width = (upper - lower) / basis
    bb_width = np.where(np.isnan(bb_width) | (basis == 0), 0, bb_width)
    
    # Bollinger Band Squeeze: width < 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr14
    minus_di = 100 * dm_minus_14 / tr14
    
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    
    # ADX: smoothed DX
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions: squeeze release + ADX trend + volume
        if squeeze[i-1] and not squeeze[i]:  # Squeeze just released
            if adx_aligned[i] > 25 and volume_filter[i]:  # Strong trend + volume
                # Breakout direction: close outside Bollinger Bands
                if close[i] > upper[i]:  # Bullish breakout
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lower[i]:  # Bearish breakout
                    signals[i] = -0.25
                    position = -1
                else:
                    # Hold current position
                    if position == 1:
                        signals[i] = 0.25
                    elif position == -1:
                        signals[i] = -0.25
                    else:
                        signals[i] = 0.0
            else:
                # Hold current position if conditions not met
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_BB_Squeeze_ADX25_Volume2x"
timeframe = "12h"
leverage = 1.0