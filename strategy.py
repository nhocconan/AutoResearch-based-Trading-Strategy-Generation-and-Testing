#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Volume Confirmation and ADX Trend Filter
Hypothesis: Price breakouts from Donchian channels with volume confirmation and
ADX trend strength capture sustained moves in both bull and bear markets.
The 1d ADX filter ensures we only trade when there is a strong trend,
reducing whipsaw in ranging markets. Volume confirmation ensures breakouts
are supported by institutional participation. This strategy targets 20-30
trades per year to minimize fee drag while capturing strong momentum moves.
"""

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
    
    # Get 1d data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr.iloc[0] = tr1.iloc[0]  # First TR is just high-low
    
    # Directional Movement
    up_move = df_1d['high'].diff()
    down_move = df_1d['low'].diff().multiply(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(series, period):
        result = np.zeros_like(series)
        if len(series) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(series[:period]) / period
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(series)):
            result[i] = result[i-1] * (1 - 1/period) + series[i] * (1/period)
        return result
    
    atr_1d = wilders_smoothing(tr.values, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels on 4h (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: current volume > 1.8x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx = adx_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        # ADX trend filter: only trade when trend is strong (ADX > 25)
        strong_trend = adx > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and strong trend
            if price > highest_high[i] and vol_ok and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and strong trend
            elif price < lowest_low[i] and vol_ok and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to Donchian mid-point or trend weakens
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if price < donchian_mid or adx < 20:  # Exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to Donchian mid-point or trend weakens
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if price > donchian_mid or adx < 20:  # Exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0