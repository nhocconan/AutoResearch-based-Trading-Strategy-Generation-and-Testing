#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_v2
Strategy: 12h price action with 1d Camarilla confluence and trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Buy when 12h closes above daily R3 with volume confirmation and daily close > prior daily close (uptrend); sell when 12h closes below daily S3 with volume confirmation and daily close < prior daily close (downtrend). Exit when price returns to daily pivot. Uses strict volume confirmation (2.0x average) to reduce trades and avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v2"
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
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h volume filter: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Close (trend filter: use prior day's close) ===
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_trend = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # === 1d Camarilla (entry levels from prior day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's Camarilla levels (use shifted values to avoid look-ahead)
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    close_1d_shift[0] = np.nan
    
    pivot_1d = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    range_1d = high_1d_shift - low_1d_shift
    r3_1d = close_1d_shift + range_1d * 1.166
    s3_1d = close_1d_shift - range_1d * 1.166
    
    # Align daily Camarilla to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(close_1d_trend[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 12h volume must be elevated (2.0x average)
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Trend filter: price close vs prior day close (1d trend)
        uptrend_1d = price_close > close_1d_trend[i]
        downtrend_1d = price_close < close_1d_trend[i]
        
        # Long conditions: 12h closes above prior day's R3 with volume + 1d uptrend
        long_signal = volume_confirmed and (price_close > r3_1d_aligned[i]) and uptrend_1d
        
        # Short conditions: 12h closes below prior day's S3 with volume + 1d downtrend
        short_signal = volume_confirmed and (price_close < s3_1d_aligned[i]) and downtrend_1d
        
        # Exit when price returns to the 1d pivot (mean reversion within prior day's range)
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

"""Hypothesis: Buy when 12h closes above daily R3 with volume confirmation and daily close > prior daily close (uptrend); sell when 12h closes below daily S3 with volume confirmation and daily close < prior daily close (downtrend). Exit when price returns to daily pivot. Uses strict volume confirmation (2.0x average) to reduce trades and avoid counter-trend trades. Designed to work in both bull and bear markets by only taking trades in direction of higher timeframe trend."""