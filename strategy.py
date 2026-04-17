#!/usr/bin/env python3
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
    
    # === 1d Williams Alligator ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_1d = (high_1d + low_1d) / 2
    
    # Williams Alligator lines (13, 8, 5 SMAs with forward shift)
    def sma(series, window, shift):
        sma_vals = np.full_like(series, np.nan)
        for i in range(len(series)):
            if i >= window - 1:
                sma_vals[i] = np.mean(series[i-window+1:i+1])
        # Apply forward shift
        shifted = np.full_like(sma_vals, np.nan)
        if shift > 0:
            shifted[shift:] = sma_vals[:-shift]
        else:
            shifted = sma_vals
        return shifted
    
    jaw = sma(median_1d, 13, 8)   # Blue line: 13-period SMA, 8 bars forward
    teeth = sma(median_1d, 8, 5)  # Red line: 8-period SMA, 5 bars forward
    lips = sma(median_1d, 5, 3)   # Green line: 5-period SMA, 3 bars forward
    
    # Align to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1d ADX (14-period) for trend strength ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close, 1)[:len(high_1d)])
    tr3 = np.abs(low_1d - np.roll(close, 1)[:len(high_1d)])
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close[0]) if len(close) > 0 else 0
    tr3[0] = np.abs(low_1d[0] - close[0]) if len(close) > 0 else 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(series, period):
        smoothed = np.full_like(series, np.nan)
        if len(series) >= period:
            smoothed[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                smoothed[i] = (smoothed[i-1] * (period-1) + series[i]) / period
        return smoothed
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h Volume confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # 20-period average volume
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    vol_confirm = volume_4h > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    warmup = 100
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        if position == 0:
            # Long: Alligator aligned up + ADX > 25 + volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                adx_aligned[i] > 25 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Alligator aligned down + ADX > 25 + volume confirmation
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  adx_aligned[i] > 25 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: Alligator lines cross (trend change)
        elif position == 1:
            # Exit long: Lips crosses below Teeth or Teeth crosses below Jaw
            if lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips crosses above Teeth or Teeth crosses above Jaw
            if lips_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ADX_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0