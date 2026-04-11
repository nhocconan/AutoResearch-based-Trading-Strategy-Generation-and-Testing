#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_breakout_v1
# Strategy: 12h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivots from 1d capture key support/resistance; breakout with volume confirms momentum; ADX filter avoids chop. Designed for low trade frequency (<30/year) to minimize fee drag in BTC/ETH.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivots, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: H4/L4, H3/L3
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h4 = close_1d + (range_1d * 1.1 / 2)
    l4 = close_1d - (range_1d * 1.1 / 2)
    h3 = close_1d + (range_1d * 1.1 / 4)
    l3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align pivots to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # 1d Volume confirmation: current volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_series = pd.Series(vol_1d)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm_1d = vol_1d > (1.5 * vol_avg_20)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d)
    
    # 1d ADX(14) for trend filter
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def _wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = _wilders_smoothing(tr, 14)
    dm_plus_14 = _wilders_smoothing(dm_plus, 14)
    dm_minus_14 = _wilders_smoothing(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX
    adx = _wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_confirm_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_h4 = close[i] > h4_aligned[i-1]
        breakdown_l4 = close[i] < l4_aligned[i-1]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Entry logic: Camarilla breakout + volume + trend
        if breakout_h4 and vol_confirm_aligned[i] and trending and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakdown_l4 and vol_confirm_aligned[i] and trending and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Camarilla level touch (H3/L3) with volume
        elif position == 1 and close[i] < l3_aligned[i] and vol_confirm_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > h3_aligned[i] and vol_confirm_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals