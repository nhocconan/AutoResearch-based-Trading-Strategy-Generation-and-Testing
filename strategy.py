#!/usr/bin/env python3
# 6h_1d_Keltner_Reversal_Strategy
# Hypothesis: On 6h timeframe, trade mean reversion at 1d Keltner Channel extremes with volume confirmation and ADX filter.
# In ranging markets (ADX < 25), price reverses at lower/upper Keltner bands; in trending markets (ADX > 25), wait for pullback to EMA20.
# Targets 50-150 total trades over 4 years by requiring confluence of band touch, volume surge, and regime filter.
# Works in both bull and bear markets due to adaptive regime filtering and mean reversion logic.

name = "6h_1d_Keltner_Reversal_Strategy"
timeframe = "6h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for Keltner middle line
    close_1d = df_1d['close'].values
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).values
    
    # Calculate 1d ATR(10) for Keltner width
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands: middle = EMA20, width = 2 * ATR(10)
    keltner_upper = ema_20 + 2 * atr_10
    keltner_lower = ema_20 - 2 * atr_10
    
    # Align 1d Keltner levels to 6h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Calculate 1d ADX for trend/ranging filter (14-period)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM using Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Ranging market (ADX < 25): mean reversion at Keltner extremes
            if adx_aligned[i] < 25:
                # Long at lower band with volume confirmation
                if (low[i] <= keltner_lower_aligned[i] * 1.002 and 
                    close[i] > keltner_lower_aligned[i] and
                    volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short at upper band with volume confirmation
                elif (high[i] >= keltner_upper_aligned[i] * 0.998 and 
                      close[i] < keltner_upper_aligned[i] and
                      volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
            # Trending market (ADX > 25): pullback to EMA20
            elif adx_aligned[i] > 25:
                # Long pullback to EMA20 in uptrend
                if (low[i] <= ema_20_aligned[i] * 1.005 and 
                    close[i] > ema_20_aligned[i] and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short pullback to EMA20 in downtrend
                elif (high[i] >= ema_20_aligned[i] * 0.995 and 
                      close[i] < ema_20_aligned[i] and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches opposite band or EMA crosses against trend
            if adx_aligned[i] < 25:
                # Ranging: exit at upper band
                if high[i] >= keltner_upper_aligned[i] * 0.998:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Trending: exit if price closes below EMA20
                if close[i] < ema_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches opposite band or EMA crosses against trend
            if adx_aligned[i] < 25:
                # Ranging: exit at lower band
                if low[i] <= keltner_lower_aligned[i] * 1.002:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Trending: exit if price closes above EMA20
                if close[i] > ema_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals