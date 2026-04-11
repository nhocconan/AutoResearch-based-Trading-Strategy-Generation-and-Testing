#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h Donchian breakout + volume confirmation + ADX trend filter.
# Uses 12h Donchian channels to identify trend direction, enters on breakouts confirmed by volume.
# ADX filter ensures we only trade in trending markets (ADX > 25), avoiding choppy conditions.
# Designed for 20-50 trades/year to minimize fee decay while capturing strong trends.
# Works in bull/bear markets by following the trend direction from higher timeframe.

name = "4h_12h_donchian_breakout_volume_adx_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high over past 20 periods
    upper_12h = np.full_like(high_12h, np.nan)
    # Lower band: lowest low over past 20 periods
    lower_12h = np.full_like(low_12h, np.nan)
    
    for i in range(19, len(high_12h)):
        upper_12h[i] = np.max(high_12h[i-19:i+1])
        lower_12h[i] = np.min(low_12h[i-19:i+1])
    
    # Calculate ADX for trend strength
    # +DM and -DM calculation
    high_diff = np.diff(high_12h, prepend=high_12h[0])
    low_diff = np.diff(low_12h, prepend=low_12h[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(np.subtract(high_12h, np.roll(high_12h, 1)))
    tr3 = np.abs(np.subtract(low_12h, np.roll(low_12h, 1)))
    tr1[0] = high_12h[0] - low_12h[0]  # First period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, 14)
    
    # Align 12h indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: 4h volume > 1.5 * 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_long = high[i] >= upper_aligned[i] and volume_filter[i] and trend_filter
        breakout_short = low[i] <= lower_aligned[i] and volume_filter[i] and trend_filter
        
        # Exit conditions: opposite Donchian touch or trend weakening
        exit_long = low[i] <= lower_aligned[i] or adx_aligned[i] < 20
        exit_short = high[i] >= upper_aligned[i] or adx_aligned[i] < 20
        
        # Entry logic: breakout in direction of trend
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals