#!/usr/bin/env python3
"""
12H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_TREND_FILTER
Hypothesis: Daily Donchian(20) breakout with volume confirmation and 1-day EMA trend filter.
In bull markets (price above 1-day EMA50), only take longs from upper band breakout.
In bear markets (price below 1-day EMA50), only take shorts from lower band breakdown.
Volume spike (2.0x 20-period) confirms institutional participation.
Target: 12-25 trades/year (50-100 total over 4 years) to stay within 12h limits.
Works in bull markets (breakouts continue with trend) and bear markets (mean reversion from extremes).
"""
name = "12H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels from prior day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    upper_band = prev_high
    lower_band = prev_low
    
    # Align Donchian levels to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # 1d EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(20, n):  # Start after warmup for volume MA
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # LONG: Price above EMA50 (bullish bias) + break above upper band + volume spike
            if (close[i] > ema50_aligned[i] and 
                close[i] > upper_band_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Price below EMA50 (bearish bias) + break below lower band + volume spike
            elif (close[i] < ema50_aligned[i] and 
                  close[i] < lower_band_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Only after minimum 4 bars AND (price re-enters range OR closes below EMA50)
            if bars_since_entry >= 4 and ((close[i] < upper_band_aligned[i] and close[i] > lower_band_aligned[i]) or 
                                          close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Only after minimum 4 bars AND (price re-enters range OR closes above EMA50)
            if bars_since_entry >= 4 and ((close[i] < upper_band_aligned[i] and close[i] > lower_band_aligned[i]) or 
                                          close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals