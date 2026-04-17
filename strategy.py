#!/usr/bin/env python3
"""
4h_12h_MultiTimeframe_VolumeBreakout_RegimeFilter
Hypothesis: Combines 12h volume surge (2x median) with 4h price breaking above 12h VWAP as momentum signal, filtered by 4h ADX>25 to avoid chop. Short when below VWAP with volume surge and ADX>25. Uses 1/3 position sizing to manage drawdown. Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    return vwap

def calculate_adx(high, low, close, period=14):
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_smooth = smooth_wilder(tr, period)
    plus_dm_smooth = smooth_wilder(plus_dm, period)
    minus_dm_smooth = smooth_wilder(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = smooth_wilder(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Data (HTF for VWAP and volume) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h VWAP
    vwap_12h = calculate_vwap(high_12h, low_12h, close_12h, volume_12h)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # 12h median volume (50-period)
    vol_median_12h = pd.Series(volume_12h).rolling(window=50, min_periods=50).median().values
    vol_median_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_12h)
    
    # 4h ADX (14-period)
    adx_4h = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_12h_aligned[i]) or 
            np.isnan(vol_median_12h_aligned[i]) or
            np.isnan(adx_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h bar's volume for surge detection
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        
        # Volume surge: current volume > 2.0x median volume
        vol_surge = vol_12h_current > 2.0 * vol_median_12h_aligned[i]
        
        # ADX filter: trending market (ADX > 25)
        trending = adx_4h[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above 12h VWAP with volume surge and trending market
            if close[i] > vwap_12h_aligned[i] and vol_surge and trending:
                signals[i] = 0.33
                position = 1
                continue
            # Short: price below 12h VWAP with volume surge and trending market
            elif close[i] < vwap_12h_aligned[i] and vol_surge and trending:
                signals[i] = -0.33
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price crosses below VWAP (momentum loss)
            if close[i] < vwap_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.33
        
        elif position == -1:
            # Exit when price crosses above VWAP (momentum loss)
            if close[i] > vwap_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.33
    
    return signals

name = "4h_12h_MultiTimeframe_VolumeBreakout_RegimeFilter"
timeframe = "4h"
leverage = 1.0