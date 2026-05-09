#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX(14) trend filter and volume confirmation.
# Donchian breakouts capture momentum in both bull and bear markets. 1w ADX > 25 filters for strong trends,
# avoiding whipsaws in ranging markets. Volume > 1.5x average confirms institutional participation.
# Designed for low trade frequency (<25/year) to minimize fee drag in bear markets.
name = "1d_Donchian20_1wADX25_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX(14) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ADX on 1w high/low/close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    def smoothed_mean(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(arr[1:period])  # skip index 0 (nan)
        # Wilder smoothing: subsequent values
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + (arr[i] if not np.isnan(arr[i]) else 0)) / period
        return result
    
    atr_1w = smoothed_mean(tr, 14)
    plus_di_1w = 100 * smoothed_mean(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * smoothed_mean(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = smoothed_mean(dx_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Donchian(20) channels from previous 1d bar
    # Upper band: max(high) over last 20 periods (excluding current)
    # Lower band: min(low) over last 20 periods (excluding current)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window:
                result[i] = np.nan
            else:
                result[i] = np.max(arr[i-window:i])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window:
                result[i] = np.nan
            else:
                result[i] = np.min(arr[i-window:i])
        return result
    
    upper = rolling_max(high, 20)
    lower = rolling_min(low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(adx_1w_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx = adx_1w_aligned[i]
        upper_band = upper[i]
        lower_band = lower[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > Upper band AND ADX > 25 (strong trend) AND volume > 1.5x average
            if close[i] > upper_band and adx > 25 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < Lower band AND ADX > 25 (strong trend) AND volume > 1.5x average
            elif close[i] < lower_band and adx > 25 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < Lower band OR ADX < 20 (trend weakening)
            if close[i] < lower_band or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > Upper band OR ADX < 20 (trend weakening)
            if close[i] > upper_band or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals