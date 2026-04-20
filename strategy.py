#!/usr/bin/env python3
# 4h_1d_Donchian_Breakout_VolumeTrend_Regime
# Hypothesis: On 4h timeframe, trade Donchian channel breakouts with 1d EMA trend filter, volume confirmation, and ADX regime filter.
# In trending markets (ADX > 25), trade breakouts in EMA direction. In ranging markets (ADX < 25), avoid trades.
# Targets 20-40 trades/year by requiring confluence of breakout, trend, volume, and regime filter.
# Works in both bull and bear markets due to trend filter and regime adaptation.

name = "4h_1d_Donchian_Breakout_VolumeTrend_Regime"
timeframe = "4h"
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
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ADX for regime filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
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
    
    atr_1d = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr_1d
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = smooth_wilder(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channel (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Only trade in trending markets (ADX > 25)
            if adx_aligned[i] > 25:
                # Long breakout above Donchian high with volume confirmation and above EMA
                if (high[i] > highest_high[i] and 
                    close[i] > ema_34_aligned[i] and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.30
                    position = 1
                # Short breakdown below Donchian low with volume confirmation and below EMA
                elif (low[i] < lowest_low[i] and 
                      close[i] < ema_34_aligned[i] and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low or trend reversal
            if low[i] < lowest_low[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: breakout above Donchian high or trend reversal
            if high[i] > highest_high[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals