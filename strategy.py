#!/usr/bin/env python3
"""
6h_Pivot_Range_Breakout_With_Trend_and_Volume
Hypothesis: Price breaking outside the 1d high-low range with 1d ADX > 25 (trending) and volume > 1.5x average captures strong momentum moves. Uses weekly pivot direction (above/below weekly pivot) to filter for higher probability trades in both bull and bear markets. Weekly pivot acts as a dynamic bias filter, reducing counter-trend trades.
"""

name = "6h_Pivot_Range_Breakout_With_Trend_and_Volume"
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
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d range (high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for breakout
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # 1d ADX for trend filter (using Wilder's smoothing)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = np.nan
        down_move[0] = np.nan
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        def WilderSmooth(data, period):
            smoothed = np.full_like(data, np.nan)
            if len(data) >= period:
                smoothed[period-1] = np.nansum(data[:period])
                for i in range(period, len(data)):
                    smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
            return smoothed
        
        tr_sum = WilderSmooth(tr, period)
        plus_dm_sum = WilderSmooth(plus_dm, period)
        minus_dm_sum = WilderSmooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_sum / tr_sum
        minus_di = 100 * minus_dm_sum / tr_sum
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    
    # Weekly pivot (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    
    # Align all 1d and 1w data to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume spike: >1.5x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(prev_high_1d[i]) or np.isnan(prev_low_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 1d high + ADX > 25 + price above weekly pivot + volume spike
            if (close[i] > prev_high_1d[i] and 
                adx_1d_aligned[i] > 25 and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 1d low + ADX > 25 + price below weekly pivot + volume spike
            elif (close[i] < prev_low_1d[i] and 
                  adx_1d_aligned[i] > 25 and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d low (range reversion)
            if close[i] < prev_low_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d high (range reversion)
            if close[i] > prev_high_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals