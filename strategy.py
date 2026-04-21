#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_With_Volume_Spike
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing reliable trend signals in both trending and ranging markets. Combined with volume spikes (2x 20-period average) and 1-day ADX > 20 for trend confirmation, this strategy captures strong moves while avoiding whipsaws. Works in bull/bear markets by following KAMA direction with volume confirmation. Targets 20-40 trades/year with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, er_period))
    change[0] = 0
    
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data once for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing
    tr_period = np.zeros_like(tr)
    dm_plus_period = np.zeros_like(dm_plus)
    dm_minus_period = np.zeros_like(dm_minus)
    
    tr_period[0] = tr[0]
    dm_plus_period[0] = dm_plus[0]
    dm_minus_period[0] = dm_minus[0]
    
    for i in range(1, len(tr)):
        tr_period[i] = tr_period[i-1] - (tr_period[i-1] / 14) + tr[i]
        dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / 14) + dm_plus[i]
        dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.where(tr_period != 0, 100 * dm_plus_period / tr_period, 0)
    di_minus = np.where(tr_period != 0, 100 * dm_minus_period / tr_period, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = np.zeros_like(dx)
    if len(dx) >= 14:
        adx_1d[13] = np.mean(dx[:14])
    
    for i in range(14, len(dx)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate KAMA on 4h close prices
    kama = calculate_kama(prices['close'].values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if ADX not ready
        if np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        kama_value = kama[i]
        
        # Volume confirmation: current volume > 2 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long: price above KAMA + volume confirmation + trending market
            if price > kama_value and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + volume confirmation + trending market
            elif price < kama_value and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA or ADX drops below 15 (losing trend)
            if price < kama_value or adx_1d_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA or ADX drops below 15 (losing trend)
            if price > kama_value or adx_1d_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Filter_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0