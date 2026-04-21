#!/usr/bin/env python3
"""
12h_PivotPoint_R1S1_Breakout_Volume_Regime
Hypothesis: On 12h timeframe, enter long when price breaks above daily pivot R1 with volume spike in trending market,
enter short when price breaks below daily pivot S1 with volume spike in trending market. Uses ADX for trend filter and ATR for volatility-based volume confirmation.
Designed for 12h with 1-2 trades per month (~24-48/year) to minimize fee drag and work in both bull/bear markets via trend-following logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([np.nan]), tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([np.array([0]), dm_plus])
    dm_minus = np.concatenate([np.array([0]), dm_minus])
    
    # Smoothed values
    atr = np.zeros_like(high)
    dm_plus_smooth = np.zeros_like(high)
    dm_minus_smooth = np.zeros_like(high)
    
    # Initial average
    atr[period] = np.nanmean(tr[1:period+1])
    dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
    dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
    
    # Wilder smoothing
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = np.zeros_like(high)
    adx[2*period] = np.nanmean(dx[period+1:2*period+1])
    
    for i in range(2*period + 1, len(high)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([np.nan]), tr])
    
    atr = np.zeros_like(high)
    atr[period] = np.nanmean(tr[1:period+1])
    
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points: P = (H + L + C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L
    r1 = 2 * pivot - low_1d
    # S1 = 2*P - H
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 1w data for ADX trend filter (higher timeframe for stronger trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on weekly data
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate ATR on weekly data for volatility-based volume threshold
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(atr_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * ATR-based threshold
        # ATR-based threshold = average volume scaled by volatility
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            vol_threshold = vol_ma * (1 + atr_1w_aligned[i] * 10)  # Scale by volatility
            volume_ok = volume > vol_threshold
        else:
            volume_ok = False
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long conditions: price breaks above R1 + volume spike + trending market
            if (price > r1_aligned[i] and 
                volume_ok and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 + volume spike + trending market
            elif (price < s1_aligned[i] and 
                  volume_ok and 
                  trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below pivot OR trend weakens
            if price < pivot_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above pivot OR trend weakens
            if price > pivot_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PivotPoint_R1S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0