#!/usr/bin/env python3
"""
6h_Adaptive_Donchian_Breakout_Volume_Regime
Hypothesis: 6h Donchian(20) breakout with volume confirmation and regime filter (ADX > 25 for trending, ADX < 20 for ranging).
In trending regime: trade breakouts in direction of 12h EMA50 trend.
In ranging regime: fade moves to Donchian bands with volume exhaustion.
Uses discrete sizing (0.25) to minimize fees. Target: 12-30 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at bands.
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
    
    # Get 6h data for Donchian calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Donchian parameters
    lookback = 20
    
    # Calculate Donchian channels for each 6h bar
    upper_6h = np.full(len(close_6h), np.nan)
    lower_6h = np.full(len(close_6h), np.nan)
    mid_6h = np.full(len(close_6h), np.nan)
    
    for i in range(lookback, len(close_6h)):
        # Use the last 20 6h bars including current
        high_max = np.max(high_6h[i-lookback+1:i+1])
        low_min = np.min(low_6h[i-lookback+1:i+1])
        
        upper_6h[i] = high_max
        lower_6h[i] = low_min
        mid_6h[i] = (high_max + low_min) / 2
    
    # Align Donchian levels to original timeframe
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    mid_6h_aligned = align_htf_to_ltf(prices, df_6h, mid_6h)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 6h data for ADX calculation (regime filter)
    # Calculate ADX components: +DM, -DM, TR
    tr = np.maximum(high_6h[1:] - low_6h[1:], np.maximum(np.abs(high_6h[1:] - close_6h[:-1]), np.abs(low_6h[1:] - close_6h[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    plus_dm = np.where((high_6h[1:] - high_6h[:-1]) > (low_6h[:-1] - low_6h[1:]), np.maximum(high_6h[1:] - high_6h[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_6h[:-1] - low_6h[1:]) > (high_6h[1:] - high_6h[:-1]), np.maximum(low_6h[:-1] - low_6h[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])  # skip first NaN
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nanmean(data[i-period+1:i+1])
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_6h = wilders_smoothing(tr, 14)
    plus_di_6h = 100 * wilders_smoothing(plus_dm, 14) / atr_6h
    minus_di_6h = 100 * wilders_smoothing(minus_dm, 14) / atr_6h
    dx_6h = 100 * np.abs(plus_di_6h - minus_di_6h) / (plus_di_6h + minus_di_6h)
    adx_6h = wilders_smoothing(dx_6h, 14)
    
    # Align ADX to original timeframe
    adx_6h_aligned = align_htf_to_ltf(prices, df_6h, adx_6h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_6h_aligned[i]) or np.isnan(lower_6h_aligned[i]) or 
            np.isnan(mid_6h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(adx_6h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime determination
        is_trending = adx_6h_aligned[i] > 25
        is_ranging = adx_6h_aligned[i] < 20
        
        if position == 0:
            if is_trending:
                # Trending regime: breakout in direction of 12h EMA50 trend
                long_signal = (close[i] > upper_6h_aligned[i]) and (close[i] > ema_50_12h_aligned[i]) and vol_spike[i]
                short_signal = (close[i] < lower_6h_aligned[i]) and (close[i] < ema_50_12h_aligned[i]) and vol_spike[i]
            else:
                # Ranging regime: fade moves to bands with volume exhaustion
                long_signal = (close[i] < lower_6h_aligned[i]) and vol_spike[i]  # oversold bounce
                short_signal = (close[i] > upper_6h_aligned[i]) and vol_spike[i]  # overbought rejection
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if is_trending:
                # In trend: exit when price crosses below mid or opposite band touched
                exit_signal = (close[i] < mid_6h_aligned[i]) or (close[i] < lower_6h_aligned[i])
            else:
                # In range: exit when price returns to mid or shows exhaustion
                exit_signal = (close[i] >= mid_6h_aligned[i]) or (close[i] >= upper_6h_aligned[i] and not vol_spike[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # In trend: exit when price crosses above mid or opposite band touched
                exit_signal = (close[i] > mid_6h_aligned[i]) or (close[i] > upper_6h_aligned[i])
            else:
                # In range: exit when price returns to mid or shows exhaustion
                exit_signal = (close[i] <= mid_6h_aligned[i]) or (close[i] <= lower_6h_aligned[i] and not vol_spike[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Adaptive_Donchian_Breakout_Volume_Regime"
timeframe = "6h"
leverage = 1.0