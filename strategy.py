#!/usr/bin/env python3
# 12h_Turtle_Soup_Reversal
# Hypothesis: Combines 12h price action with 1d Turtle Soup reversal pattern (false breakout of 20-period high/low).
# Goes long when price makes a new 20-period low then closes back above it (bull trap).
# Goes short when price makes a new 20-period high then closes back below it (bear trap).
# Uses 1d ADX to filter ranging markets (ADX < 20) and volume confirmation to avoid false signals.
# Designed for low frequency (<20 trades/year) to minimize fee drag, works in both trending and ranging markets.

name = "12h_Turtle_Soup_Reversal"
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
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h indicators: 20-period highest high and lowest low
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in low ADX (ranging) markets
        if adx_1d_aligned[i] >= 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long Turtle Soup: new 20L then close back above it (bull trap)
            if (low[i] <= lowest_20[i] and 
                close[i] > lowest_20[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short Turtle Soup: new 20H then close back below it (bear trap)
            elif (high[i] >= highest_20[i] and 
                  close[i] < highest_20[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below the 20-period low (stop and reverse)
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above the 20-period high (stop and reverse)
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals