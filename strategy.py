#!/usr/bin/env python3
"""
12h_1w_camarilla_breakout_volume
Strategy: 12h breakout with weekly Camarilla levels and volume confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: Buy when 12h closes above prior weekly R3 with volume expansion; sell when 12h closes below prior weekly S3 with volume expansion. Exit when price returns to weekly pivot. Uses weekly levels for structural context, reducing whipsaw in sideways markets. Designed for both bull and bear markets by focusing on volatility breakouts from key weekly levels rather than trend direction. Low-frequency design targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_volume"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Weekly Close (prior close for context) ===
    close_1w = df_1w['close'].values
    close_1w_shifted = np.roll(close_1w, 1)
    close_1w_shifted[0] = np.nan
    close_1w_prior = align_htf_to_ltf(prices, df_1w, close_1w_shifted)
    
    # === Weekly High/Low/Close (shifted for prior week) ===
    high_1w_shift = np.roll(df_1w['high'].values, 1)
    low_1w_shift = np.roll(df_1w['low'].values, 1)
    close_1w_shift = np.roll(close_1w, 1)
    high_1w_shift[0] = np.nan
    low_1w_shift[0] = np.nan
    close_1w_shift[0] = np.nan
    
    # === Weekly Camarilla (from prior week) ===
    pivot_1w = (high_1w_shift + low_1w_shift + close_1w_shift) / 3
    range_1w = high_1w_shift - low_1w_shift
    r3_1w = close_1w_shift + range_1w * 1.166
    s3_1w = close_1w_shift - range_1w * 1.166
    
    # Align weekly Camarilla to 12h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Session filter: 00-23 UTC (12h captures full day, but avoid illiquid hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(close_1w_prior[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 12h volume must be expanded
        volume_expanded = volume_current > 1.5 * vol_ma
        
        # Long conditions: 12h closes above prior weekly R3 with volume expansion
        long_signal = volume_expanded and (price_close > r3_1w_aligned[i])
        
        # Short conditions: 12h closes below prior weekly S3 with volume expansion
        short_signal = volume_expanded and (price_close < s3_1w_aligned[i])
        
        # Exit when price returns to the weekly pivot (mean reversion within prior week's range)
        exit_long = position == 1 and price_close < pivot_1w_aligned[i]
        exit_short = position == -1 and price_close > pivot_1w_aligned[i]
        
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

# Hypothesis: Buy when 12h closes above prior weekly R3 with volume expansion; sell when 12h closes below prior weekly S3 with volume expansion. Exit when price returns to weekly pivot. Uses weekly levels for structural context, reducing whipsaw in sideways markets. Designed for both bull and bear markets by focusing on volatility breakouts from key weekly levels rather than trend direction. Low-frequency design targets 15-30 trades/year to minimize fee drag.