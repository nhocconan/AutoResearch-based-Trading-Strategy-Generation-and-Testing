#!/usr/bin/env python3
"""
6h_1w_1d_adx_volume_breakout_v1
Strategy: 6h Donchian breakout filtered by 1-week ADX trend strength and 1-day volume spike
Timeframe: 6h
Leverage: 1.0
Hypothesis: In trending markets (ADX > 25 on weekly), price breaks of 6-hour Donchian channels (20-period) with volume confirmation (>1.5x average) capture sustained moves. Weekly ADX ensures we only trade in strong trends, avoiding chop. Volume confirms breakout legitimacy. Designed to work in both bull (uptrend breaks) and bear (downtrend breaks) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_adx_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly ADX (14-period) for trend strength ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip first NaN in tr
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1w = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w > 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w > 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = wilder_smooth(dx, 14)
    
    # Align ADX to 6h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === Daily volume average (20-period) ===
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # === 6-hour Donchian channels (20-period) ===
    def donchian_channels(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_high, donch_low = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Trend filter: strong trend (ADX > 25)
        strong_trend = adx_1w_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x daily average
        vol_confirmed = volume[i] > 1.5 * vol_avg_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = price_high > donch_high[i]  # Using high to capture breakout
        breakout_down = price_low < donch_low[i]   # Using low to capture breakdown
        
        # Exit conditions: price returns to opposite Donchian level or ADX weakens
        exit_long = position == 1 and (price_low < donch_low[i] or adx_1w_aligned[i] < 20)
        exit_short = position == -1 and (price_high > donch_high[i] or adx_1w_aligned[i] < 20)
        
        # Trading logic
        if breakout_up and vol_confirmed and strong_trend and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_confirmed and strong_trend and position != -1:
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