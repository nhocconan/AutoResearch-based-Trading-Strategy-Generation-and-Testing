#!/usr/bin/env python3
"""
1d Weekly ATR Breakout with Volume Spike and Trend Filter
Hypothesis: Weekly ATR-based breakouts capture strong momentum moves. 
Volume surge confirms institutional participation. Trend filter (price vs weekly EMA20) 
ensures alignment with higher timeframe direction. Works in both bull and bear 
markets by following breakout direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    ema = np.zeros_like(arr)
    multiplier = 2 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = (arr[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly ATR and EMA20
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_atr = calculate_atr(weekly_high, weekly_low, weekly_close, 14)
    weekly_ema20 = calculate_ema(weekly_close, 20)
    
    # Align to daily timeframe (use previous week's values)
    weekly_atr_aligned = align_htf_to_ltf(prices, df_1w, weekly_atr)
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Weekly breakout levels: previous week close ± ATR * multiplier
    weekly_close_prev = np.roll(weekly_close, 1)
    weekly_close_prev[0] = weekly_close[0]  # First value
    breakout_up = weekly_close_prev + (weekly_atr_aligned * 1.5)
    breakout_down = weekly_close_prev - (weekly_atr_aligned * 1.5)
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma20[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Trend filter: price above/below weekly EMA20
    price_above_ema = close > weekly_ema20_aligned
    price_below_ema = close < weekly_ema20_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(vol_ma20[i]) or np.isnan(weekly_ema20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly resistance with volume spike and uptrend
            if (close[i] > breakout_up[i] and 
                vol_spike[i] and 
                price_above_ema[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly support with volume spike and downtrend
            elif (close[i] < breakout_down[i] and 
                  vol_spike[i] and 
                  price_below_ema[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below breakout level or volume spike ends
            if close[i] < breakout_up[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above breakout level or volume spike ends
            if close[i] > breakout_down[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_ATRBreakout_VolumeSpike_TrendFilter"
timeframe = "1d"
leverage = 1.0