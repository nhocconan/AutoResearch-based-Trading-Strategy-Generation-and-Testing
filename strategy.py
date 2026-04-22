#!/usr/bin/env python3
"""
Hypothesis: 1-hour ADX-based trend strength with 4-hour RSI overbought/oversold and volume confirmation.
Long when ADX > 25 (strong trend), RSI(4h) < 30 (oversold), and volume > 1.5x 20-period average.
Short when ADX > 25, RSI(4h) > 70 (overbought), and volume > 1.5x 20-period average.
Exit when ADX < 20 (weakening trend) or RSI(4h) returns to neutral zone (40-60).
Uses 4h for trend/filter conditions, 1h only for entry timing. Designed for low trade frequency by requiring
trend strength + extreme RSI + volume confirmation. Works in bull markets (buy oversold in uptrend) and
bear markets (sell overbought in downtrend). ADX filter prevents whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4-hour data for ADX and RSI - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    # Calculate ADX(14) on 4h
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def _wilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = _wilder_smoothing(tr, 14)
    dm_plus_smooth = _wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = _wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = _wilder_smoothing(dx, 14)
    
    # Calculate RSI(14) on 4h
    def _rsi(series, period=14):
        delta = np.diff(series)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().values
        roll_down = pd.Series(down).ewm(alpha=1/period, adjust=False).mean().values
        rs = np.where(roll_down != 0, roll_up / roll_down, 0)
        rsi = 100 - (100 / (1 + rs))
        # Pad first value
        return np.concatenate([[np.nan], rsi])
    
    rsi_4h = _rsi(close_4h, 14)
    
    # Align to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Strong trend (ADX > 25), RSI oversold (< 30), volume spike
            if (adx_4h_aligned[i] > 25 and 
                rsi_4h_aligned[i] < 30 and vol_spike):
                signals[i] = 0.20
                position = 1
            # Short: Strong trend (ADX > 25), RSI overbought (> 70), volume spike
            elif (adx_4h_aligned[i] > 25 and 
                  rsi_4h_aligned[i] > 70 and vol_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Weakening trend (ADX < 20) OR RSI returns to neutral (40-60)
            exit_signal = False
            
            if adx_4h_aligned[i] < 20 or (40 <= rsi_4h_aligned[i] <= 60):
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_ADX_RSI_Trend_Strength_Volume"
timeframe = "1h"
leverage = 1.0