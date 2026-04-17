#!/usr/bin/env python3
"""
6h_FibPivot_R1S1_EMA34_VolumeSpike_ATRFilter
Strategy: 6h Fibonacci pivot breakout with EMA34 filter, volume spike, and ATR volatility filter.
Long: Close > R1 + EMA34 + volume > 1.5x 20-bar avg + ATR(14) > 0.5x ATR(50)
Short: Close < S1 + EMA34 + volume > 1.5x 20-bar avg + ATR(14) > 0.5x ATR(50)
Exit: Close crosses EMA34 in opposite direction
Position size: 0.25
Uses Fibonacci pivots from daily timeframe for key levels, EMA34 for trend,
volume spike for conviction, ATR filter to avoid low volatility whipsaws.
Works in both bull and bear markets by fading at key daily levels with confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # first period
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr.values

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Fibonacci pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Fibonacci pivot points (daily)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + 0.382 * range_1d
    s1 = pivot - 0.382 * range_1d
    r2 = pivot + 0.618 * range_1d
    s2 = pivot - 0.618 * range_1d
    
    # Calculate EMA34 on daily close
    ema34_1d = calculate_ema(close_1d, 34)
    
    # Calculate ATR on 6h data
    atr_6h = calculate_atr(high, low, close, 14)
    atr_ma50_6h = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-period volume average
    vol_ma20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(60, n):  # warmup for calculations
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isclose(r1_aligned[i], 0) or np.isclose(s1_aligned[i], 0) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(atr_ma50_6h[i]) or np.isnan(vol_ma20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-bar average
        volume_filter = volume[i] > (1.5 * vol_ma20_6h[i])
        
        # ATR filter: current ATR > 0.5x 50-bar average ATR (avoid low volatility)
        atr_filter = atr_6h[i] > (0.5 * atr_ma50_6h[i])
        
        # Trend filter: price vs daily EMA34
        above_ema34 = close[i] > ema34_1d_aligned[i]
        below_ema34 = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: break above R1 + above EMA34 + volume spike + ATR filter
            if (close[i] > r1_aligned[i] and above_ema34 and 
                volume_filter and atr_filter):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + below EMA34 + volume spike + ATR filter
            elif (close[i] < s1_aligned[i] and below_ema34 and 
                  volume_filter and atr_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA34
            if below_ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34
            if above_ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_FibPivot_R1S1_EMA34_VolumeSpike_ATRFilter"
timeframe = "6h"
leverage = 1.0