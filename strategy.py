#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_momentum_v1
Strategy: 12h breakout of prior day's Camarilla levels with momentum confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses prior day's Camarilla S3/R3 levels as breakout levels, confirmed by 12h price momentum (close > open for longs, close < open for shorts) and volume expansion (>1.5x 20-period average). Only trades during active sessions (08-20 UTC). Designed for low trade frequency (~20-40/year) to avoid fee drag while capturing momentum moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d OHLC (prior day) ===
    high_1d_shift = np.roll(df_1d['high'].values, 1)
    low_1d_shift = np.roll(df_1d['low'].values, 1)
    close_1d_shift = np.roll(df_1d['close'].values, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    # Prior day's Camarilla levels
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d * 1.166  # Camarilla R3
    s3_1d = close_1d_shift - range_1d * 1.166  # Camarilla S3
    
    # Align 1d Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_open = open_price[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: expanded volume (>1.5x 20-period average)
        volume_expanded = volume_current > 1.5 * vol_ma
        
        # Momentum confirmation: strong directional candle
        bullish_momentum = price_close > price_open
        bearish_momentum = price_close < price_open
        
        # Long conditions: price breaks above prior day's R3 with volume expansion + bullish momentum
        long_signal = volume_expanded and bullish_momentum and (price_close > r3_1d_aligned[i])
        
        # Short conditions: price breaks below prior day's S3 with volume expansion + bearish momentum
        short_signal = volume_expanded and bearish_momentum and (price_close < s3_1d_aligned[i])
        
        # Exit when price returns to prior day's pivot level (mean reversion)
        exit_long = position == 1 and price_close < pivot_1d_aligned[i]
        exit_short = position == -1 and price_close > pivot_1d_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals